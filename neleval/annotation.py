#!/usr/bin/env python
"Representation of link standoff annotation and measures over it"

from __future__ import division, print_function
from collections import Sequence, defaultdict
import operator
import warnings
import json

from .utils import unicode


try:
    keys = dict.viewkeys
    import itertools
    filter = itertools.ifilter
except Exception:
    # Py3k
    keys = dict.keys


class Annotation(object):
    """A name mention and its predicted candidates

    Parameters
    ----------

    Attributes
    ----------
    TODO
    is_first : bool or unset
        Indicates if this annotation is the first in the document for this
        eid.

    * :
        Other attributes are automatically adopted from the top candidate
    """

    __slots__ = ['docid', 'start', 'end', 'candidates', 'is_first']

    def __init__(self, docid, start, end, candidates=[]):
        self.docid = docid
        self.start = start
        self.end = end
        self.candidates = candidates

    def __unicode__(self):
        return u'{}\t{}\t{}\t{}'.format(
            self.docid,
            self.start,
            self.end,
            u'\t'.join([unicode(c) for c in self.candidates])
            )

    __str__ = __unicode__

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

    def __getattr__(self, name):
        # Generally, return attributes from top candidate
        # or None if there are no candidates
        if name.startswith('_'):
            # AttributeError
            super(Annotation, self).__getattr__(name)
        link = self.link
        if link is not None:
            return getattr(link, name)

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
        return [f for f in dir(ann) + dir(ann.link)
                if not f.startswith('_')
                and not callable(getattr(ann, f, None))]


class Candidate(object):
    """A candidate link

    Parameters
    ----------
    eid : string
        KB ID or NIL ID (begins "NIL")
    score : numeric
        Confidence
    type : string or dict
        If a string, the ``type`` attribute is set to this string.
        If a dict, all keys are copied to attributes of the candidate

    Attributes
    ----------
    eid : string
    score : numeric
    kbid : string or None
        ``eid`` if it does not begin with "NIL"
    is_nil : bool
        True if ``eid`` begins with "NIL"
    is_linked : bool
        ``not is_nil``

    Examples
    --------
    >>> cand = Candidate.from_string('NIL123')
    >>> (cand.eid, cand.score, cand.kbid, cand.is_nil, cand.is_linked)
    ('NIL123', None, None, True, False)
    >>> cand = Candidate.from_string('E123\t1.5\tPER')
    >>> (cand.eid, cand.score, cand.kbid, cand.is_nil, cand.is_linked)
    ('E123', 1.5, 'E123', False, True)
    >>> cand.type
    'PER'
    >>> cand = Candidate.from_string('E123\t1.5\t{"type":"PER","reftype":"NOM"}')
    >>> cand.type
    'PER'
    >>> cand.reftype
    'NOM'
    """
    __slots__ = ['eid',    # KB ID or NIL ID
                 'score',  # confidence
                 '__dict__',
                 ]

    def __init__(self, eid, score=None, type=None):
        self.eid = eid
        self.score = score
        if hasattr(type, 'items'):
            self.__dict__.update(type)
        else:
            self.type = type

    @property
    def kbid(self):
        "Return link KB ID or None"
        if self.is_linked:
            return self.eid

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

    def __unicode__(self):
        type_ = self.__dict__
        if not type_:
            type_ = ''
        elif len(type_) == 1 and 'type' in type_:
            type_ = type_['type'] or ''
        else:
            type_ = json.dumps(type_)
        return u'{}\t{}\t{}'.format(self.eid,
                                    self.score or '',
                                    type_)

    __str__ = __unicode__

    def __repr__(self):
        return '<{!r}>'.format(self.eid)

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
                if type.startswith('{'):
                    type = json.loads(type)
                yield cls(id, float(score), type)
        else:
            # undefined format
            raise SyntaxError('Need id, score and type when >1 candidates')


class Measure(object):
    def __init__(self, key, filter=None, agg='sets', weighting=None):
        """
        key : list of fields for mention comparison
        filter : a function or attribute name to select evaluated annotations
        agg : sets or clustering measure
        weighting : mapping field -> (func(gold_val, pred_val) -> float)
        """
        if not isinstance(key, Sequence):
            raise TypeError('key should be a list or tuple')
        self.key = tuple(key)
        self.filter = filter
        if filter is not None and not callable(filter):
            assert isinstance(filter, str)
            filter = operator.attrgetter(filter)
        self.filter_fn = filter
        if agg.endswith('-micro'):
            self.display_agg = agg
            agg = agg[:-6]
            warnings.warn('`{0}-micro\' aggregate has been renamed to '
                          '`{0}\' and will be removed in a future '
                          'release.'.format(agg),
                          DeprecationWarning)

        self.agg = agg

        if agg != 'sets' and weighting:
            raise NotImplementedError('weighting is only implemented for '
                                      'aggregate="sets"')
        self.weighting = weighting

    def with_weighting(self, weighting):
        return Measure(self.key, self.filter, self.agg, weighting)

    def __str__(self):
        return '{}:{}:{}'.format(getattr(self, 'display_agg', self.agg),
                                 self.filter, '+'.join(self.key))

    @classmethod
    def from_string(cls, s, weighting=None):
        if s.count(':') != 2:
            raise ValueError('Expected 2 colons in {!r}'.format(s))
        a, f, k = s.split(':')
        if f in ('', 'None'):
            f = None
        return cls(k.split('+'), f, a, weighting=weighting)

    def __repr__(self):
        return ('{0.__class__.__name__}('
                '{0.key!r}, {0.filter!r}, {0.agg!r})'.format(self))

    NON_CLUSTERING_AGG = ('sets',) + tuple(
        ['overlap-%s%s' % (p1, p2)
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
        if not self.weighting:
            gold_index = self.build_index(gold)
            pred_index = self.build_index(system)
            tp = len(keys(gold_index) & keys(pred_index))

            fn = len(gold_index) - tp
            fp = len(pred_index) - tp
        else:
            key = [f for f in self.key
                   if f not in self.weighting]
            weighting_fields = [f for f in self.key
                                if f in self.weighting]
            gold_index = self.build_index(gold, key_fields=key, multi=True)
            pred_index = self.build_index(system, key_fields=key, multi=True)
            if (
                    any(len(v) > 1 for v in gold_index.values()) or
                    any(len(v) > 1 for v in pred_index.values())
            ):
                raise NotImplementedError('No weighting support where '
                                          'annotations may have duplicate key')
            tp = 0.
            for k, (gold_ann,) in gold_index.items():
                try:
                    pred_ann, = pred_index[k]
                except KeyError:
                    continue
                tp += self.calc_match_weight(gold_ann, pred_ann,
                                             weighting_fields)

            fn = sum(self.calc_match_weight(gold_ann, gold_ann, weighting_fields)
                     for gold_ann, in gold_index.values()) - tp
            fp = sum(self.calc_match_weight(pred_ann, pred_ann, weighting_fields)
                     for pred_ann, in pred_index.values()) - tp
        return tp, fp, fn

    def calc_match_weight(self, gold_ann, pred_ann, weighting_fields):
        w = 1
        for field in weighting_fields:
            w *= self.weighting[field](getattr(gold_ann, field),
                                       getattr(pred_ann, field))
        return w


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
        if self.weighting:
            raise NotImplementedError('get_matches not implemented with '
                                      'weighting')
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
        fp = len(overlaps_sys) - self.measure_overlap(overlaps_sys, sys_mode)
        fn = len(overlaps_gold) - self.measure_overlap(overlaps_gold, gold_mode)
        return fp, fn

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
        elif self.agg == 'sets':
            tp, fp, fn = self.count_matches(system, gold)
            return tp, fp, tp, fn
        elif self.agg.startswith('overlap-'):
            params = self.agg[len('overlap-'):]
            fp, fn = self.count_overlap(system, gold,
                                        params[:3], params[3:])
            return len(system) - fp, fp, len(gold) - fn, fn
        else:
            # This should not be reachable
            raise ValueError('Unexpected value for agg: %r' % self.agg)
