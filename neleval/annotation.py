#!/usr/bin/env python
"Representation of link standoff annotation and measures over it"

from __future__ import division, print_function
from collections import Sequence, defaultdict
import operator
from fractions import Fraction

from .utils import unicode


try:
    keys = dict.viewkeys
    import itertools
    filter = itertools.ifilter
except Exception:
    # Py3k
    keys = dict.keys


class Annotation(object):
    __slots__ = ['docid', 'start', 'end', 'candidates', 'is_first']

    def __init__(self, docid, start, end, candidates=[]):
        self.docid = docid
        self.start = start
        self.end = end
        self.candidates = candidates

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        return u'{}\t{}\t{}\t{}'.format(
            self.docid,
            self.start,
            self.end,
            u'\t'.join([unicode(c) for c in self.candidates])
            )

    def __repr__(self):
        return 'Annotation({!r}, {!r}, {!r}, {!r})'.format(self.docid, self.start, self.end, self.candidates)

    def __lt__(self, other):
        assert isinstance(other, Annotation)
        return (self.start, -self.end) < (other.start, -other.end)

    def compare_spans(self, other):
        assert self.start <= self.end, 'End is before start: {!r}'.format(self)
        assert other.start <= other.end, 'End is before start: {!r}'.format(self)
        if self.docid != other.docid:
            return 'different documents'
        if self.start > other.end or self.end < other.start:
            return 'non-overlapping'
        elif self.start == other.start and self.end == other.end:
            return 'duplicate'
        elif self.start < other.start and self.end >= other.end:
            return 'nested'
        elif self.start >= other.start and self.end < other.end:
            return 'nested'
        else:
            return 'crossing'

    # Getters
    @property
    def span(self):
        return (self.docid, self.start, self.end)

    @property
    def link(self):
        "Return top candidate"
        if self.candidates:
            return self.candidates[0]

    @property
    def eid(self):
        "Return link KB ID or NIL cluster ID (default cluster ID is None)"
        if self.link is not None:
            return self.link.id

    @property
    def kbid(self):
        "Return link KB ID or None"
        if self.is_linked:
            return self.link.id

    @property
    def score(self):
        "Return link score or None"
        if self.is_linked:
            return self.link.score

    @property
    def type(self):
        "Return link type or None"
        if self.link:
            return self.link.type

    @property
    def is_nil(self):
        if self.eid is None:
            return True
        if self.eid.startswith('NIL'):
            return True
        return False

    @property
    def is_linked(self):
        return not self.is_nil

    # Parsing methods
    @classmethod
    def from_string(cls, s):
        docid, start, end, candidates = None, None, None, []
        cols = s.rstrip('\n\t').split('\t', 3)
        if len(cols) < 3:
            raise SyntaxError('Annotation must have at least 3 columns. Got {!r}'.format(s))
        if len(cols) >= 3:
            docid = cols[0]
            start = int(cols[1])
            end = int(cols[2])
        if len(cols) == 4:
            candidates = sorted(Candidate.from_string(cols[3]), reverse=True)
        return Annotation(docid, start, end, candidates)

    @classmethod
    def list_fields(cls):
        ann = cls.from_string('a\t0\t1\tabc')
        return [f for f in dir(ann)
                if not f.startswith('_')
                and not callable(getattr(ann, f, None))]


class Candidate(object):
    __slots__ = ['id', 'score', 'type']

    def __init__(self, id, score=None, type=None):
        self.id = id
        self.score = score
        self.type = type

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        return u'{}\t{}\t{}'.format(self.id,
                                    self.score or '',
                                    self.type or '')

    def __repr__(self):
        return '<{!r}>'.format(self.id)

    def __lt__(self, other):
        assert isinstance(other, Candidate)
        return self.score < other.score

    # Parsing methods
    @classmethod
    def from_string(cls, s):
        cols = s.rstrip('\t').split('\t')
        if len(cols) == 1:
            # link includes id only
            yield cls(cols[0])
        elif len(cols) == 2:
            # link includes id and score
            yield cls(cols[0], float(cols[1]))
        elif len(cols[3:]) % 3 == 0:
            # >=1 (id, score, type) candidate tuples
            for i in range(0, len(cols), 3):
                id, score, type = cols[i:i+3]
                yield cls(id, float(score), type)
        else:
            # undefined format
            raise SyntaxError('Need id, score and type when >1 candidates')


class Measure(object):
    __slots__ = ['key', 'filter', 'filter_fn', 'agg']

    def __init__(self, key, filter=None, agg='sets-micro'):
        """
        key : list of fields for mention comparison
        filter : a function or attribute name to select evaluated annotations
        agg : [work in progress]
        """
        if not isinstance(key, Sequence):
            raise TypeError('key should be a list or tuple')
        self.key = tuple(key)
        self.filter = filter
        if filter is not None and not callable(filter):
            assert isinstance(filter, str)
            filter = operator.attrgetter(filter)
        self.filter_fn = filter
        self.agg = agg

    def __str__(self):
        return '{}:{}:{}'.format(self.agg, self.filter, '+'.join(self.key))

    @classmethod
    def from_string(cls, s):
        if s.count(':') != 2:
            raise ValueError('Expected 2 colons in {!r}'.format(s))
        a, f, k = s.split(':')
        if f in ('', 'None'):
            f = None
        return cls(k.split('+'), f, a)

    def __repr__(self):
        return ('{0.__class__.__name__}('
                '{0.key!r}, {0.filter!r}, {0.agg!r})'.format(self))

    # TODO: variants macro-averaged across docs
    NON_CLUSTERING_AGG = (['sets-micro'] +
                          ['overlap-%s%s-micro' % (p1, p2)
                           for p1 in ('sum', 'max')
                           for p2 in ('sum', 'max')])

    @property
    def is_clustering(self):
        return self.agg not in self.NON_CLUSTERING_AGG

    def build_index(self, annotations, key_fields=None, multi=False):
        if isinstance(annotations, dict):
            # assume already built
            return annotations
        # TODO: caching

        if self.filter is not None:
            annotations = filter(self.filter_fn, annotations)
        key = self.key if key_fields is None else key_fields
        if multi:
            out = defaultdict(list)
            for ann in annotations:
                out[tuple(getattr(ann, field) for field in key)].append(ann)
            out.default_factory = None
            return out
        else:
            return {tuple(getattr(ann, field) for field in key): ann
                    for ann in annotations}

    def build_clusters(self, annotations):
        if isinstance(annotations, dict):
            # assume already built
            return annotations
        # TODO: caching
        # TODO: can reuse build_index for small efficiency loss

        if self.filter is not None:
            annotations = filter(self.filter_fn, annotations)
        key = self.key
        out = defaultdict(set)
        for ann in annotations:
            out[ann.eid].add(tuple(getattr(ann, field) for field in key))
        out.default_factory = None  # disable defaulting
        return out

    def count_matches(self, system, gold):
        if self.is_clustering:
            raise ValueError('count_matches is inappropriate '
                             'for {}'.format(self.agg))
        gold_index = self.build_index(gold)
        pred_index = self.build_index(system)
        tp = len(keys(gold_index) & keys(pred_index))
        fn = len(gold_index) - tp
        fp = len(pred_index) - tp
        return tp, fp, fn

    def get_matches(self, system, gold):
        """ Assesses the match between sets of annotations

        Returns three lists of items:
        * tp [(item, other_item), ...]
        * fp [(None, other_item), ...]
        * fn [(item, None), ...]
        """
        if self.is_clustering:
            raise ValueError('get_matches is inappropriate '
                             'for {}'.format(self.agg))
        gold_index = self.build_index(gold)
        pred_index = self.build_index(system)
        gold_keys = keys(gold_index)
        pred_keys = keys(pred_index)
        shared = gold_keys & pred_keys
        tp = [(gold_index[k], pred_index[k]) for k in shared]
        fp = [(None, pred_index[k]) for k in pred_keys - shared]
        fn = [(gold_index[k], None) for k in gold_keys - shared]
        return tp, fp, fn

    def get_overlapping(self, system, gold):
        """
        Returns: overlaps_sys, overlaps_gold
            where each is a dict of sys/gold annotations mapped to a list
            of sorted annotations that overlap and have the same key apart
            from span.
        """
        key_fields = self.key
        if 'span' in key_fields:
            key_fields = [f for f in key_fields if f != 'span'] + ['docid']

        overlaps_sys = {ann: [] for ann in system}
        overlaps_gold = {ann: [] for ann in gold}
        system = self.build_index(system, multi=True, key_fields=key_fields)
        gold = self.build_index(gold, multi=True, key_fields=key_fields)
        for key, sys_annots in system.items():
            gold_annots = gold.pop(key, [])
            sys_annots.sort()
            gold_annots.sort()
            while sys_annots and gold_annots:
                rel = sys_annots[-1].compare_spans(gold_annots[-1])
                if rel != 'non-overlapping':
                    overlaps_sys[sys_annots[-1]].append(gold_annots[-1])
                    overlaps_gold[gold_annots[-1]].append(sys_annots[-1])
                if sys_annots[-1] > gold_annots[-1]:
                    sys_annots.pop()
                else:
                    gold_annots.pop()

        return ({k: list(reversed(v)) for k, v in overlaps_sys.items()},
                {k: list(reversed(v)) for k, v in overlaps_gold.items()})

    @staticmethod
    def measure_overlap(overlaps, mode):
        total = 0.
        if mode == 'sum':
            for ref, cands in overlaps.items():
                if not cands:
                    continue
                # XXX: cands should not be overlapping, but just in case...
                n_chars = 0
                opened = None
                n_open = 0
                offsets = ([(ann.start, '(') for ann in cands] +
                           [(ann.end, ')') for ann in cands])
                offsets.sort()
                for o, d in offsets:  # sorted
                    if d == '(':
                        if opened is None:
                            opened = max(o, ref.start)
                            n_open += 1
                        else:
                            n_open += 1
                    else:
                        assert n_open
                        n_open -= 1
                        if not n_open:
                            n_chars += min(o, ref.end) - opened + 1
                            opened = None
                assert n_open == 0
                assert opened is None
                total += n_chars / (ref.end - ref.start + 1)

        elif mode == 'max':
            for ref, cands in overlaps.items():
                if not cands:
                    continue
                start = ref.start
                end = ref.end
                most = max(min(end, ann.end) - max(start, ann.start)
                           for ann in cands)
                total += (most + 1) / (end - start + 1)
        else:
            raise ValueError('Unexpected overlap measurement mode: %r' % mode)

        return total

    def count_overlap(self, system, gold, gold_mode='sum', sys_mode='sum'):
        # XXX: note by convention modes are gold then sys!
        overlaps_sys, overlaps_gold = self.get_overlapping(system, gold)
        tp = self.measure_overlap(overlaps_gold, gold_mode)
        fp = len(overlaps_sys) - self.measure_overlap(overlaps_sys, sys_mode)
        fn = len(overlaps_gold) - tp
        return tp, fp, fn

    def count_clustering(self, system, gold):
        from . import coref_metrics
        if not self.is_clustering:
            raise ValueError('evaluate_clustering is inappropriate '
                             'for {}'.format(self.agg))
        try:
            fn = getattr(coref_metrics, self.agg)
        except AttributeError:
            raise ValueError('Invalid aggregation: {!r}'.format(self.agg))
        if not callable(fn):
            raise ValueError('Invalid aggregation: {!r}'.format(self.agg))
        gold_clusters = self.build_clusters(gold)
        pred_clusters = self.build_clusters(system)
        return fn(gold_clusters, pred_clusters)

    def contingency(self, system, gold):
        if self.is_clustering:
            p_num, p_den, r_num, r_den = self.count_clustering(system, gold)
            ptp = p_num
            fp = p_den - p_num
            rtp = r_num
            fn = r_den - r_num
            return ptp, fp, rtp, fn
        elif self.agg == 'sets-micro':
            tp, fp, fn = self.count_matches(system, gold)
            return tp, fp, tp, fn
        elif self.agg.startswith('overlap-') and self.agg.endswith('-micro'):
            params = self.agg[len('overlap-'):-len('-micro')]
            tp, fp, fn = self.count_overlap(system, gold,
                                            params[:3], params[3:])
            return tp, fp, tp, fn
        else:
            # This should not be reachable
            raise ValueError('Unexpected value for agg: %r' % self.agg)

    def docs_to_contingency(self, system, gold):
        return self.contingency([a for doc in system for a in doc.annotations],
                                [a for doc in gold for a in doc.annotations])
