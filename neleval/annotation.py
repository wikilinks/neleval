#!/usr/bin/env python
"Representation of link standoff annotation and matching over it"

from collections import Sequence, defaultdict
import operator


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
        return unicode(self)

    def __unicode__(self):
        return u'{}\t{}\t{}\t{}'.format(
            self.docid,
            self.start,
            self.end,
            u'\t'.join([unicode(c) for c in self.candidates])
            )

    def __cmp__(self, other):
        assert isinstance(other, Annotation)
        return cmp((self.start, -self.end), (other.start, -other.end))

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
            raise SyntaxError('Annotation must have at least 3 columns')
        if len(cols) >= 3:
            docid = cols[0]
            start = int(cols[1])
            end = int(cols[2])
        if len(cols) == 4:
            candidates = sorted(Candidate.from_string(cols[3]), reverse=True)
        return Annotation(docid, start, end, candidates)


class Candidate(object):
    __slots__ = ['id', 'score', 'type']

    def __init__(self, id, score=None, type=None):
        self.id = id
        self.score = score
        self.type = type

    def __str__(self):
        return unicode(self)

    def __unicode__(self):
        return u'{}\t{}\t{}'.format(self.id,
                                    self.score or '',
                                    self.type or '')

    def __cmp__(self, other):
        assert isinstance(other, Candidate)
        return cmp(self.score, other.score)

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
            for i in xrange(0, len(cols), 3):
                id, score, type = cols[i:i+3]
                yield cls(id, float(score), type)
        else:
            # undefined format
            raise SyntaxError('Need id, score and type when >1 candidates')


class Matcher(object):
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

    NON_CLUSTERING_AGG = ('sets-micro',)  # 'sets-macro')

    @property
    def is_clustering_match(self):
        return self.agg not in self.NON_CLUSTERING_AGG

    def build_index(self, annotations):
        if isinstance(annotations, dict):
            # assume already built
            return annotations
        # TODO: caching

        if self.filter is not None:
            annotations = filter(self.filter_fn, annotations)
        key = self.key
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
        if self.is_clustering_match:
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
        if self.is_clustering_match:
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

    def count_clustering(self, system, gold):
        from . import coref_metrics
        if not self.is_clustering_match:
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
        if self.is_clustering_match:
            p_num, p_den, r_num, r_den = self.count_clustering(system, gold)
            ptp = p_num
            fp = p_den - p_num
            rtp = r_num
            fn = r_den - r_num
            return ptp, fp, rtp, fn
        else:
            tp, fp, fn = self.count_matches(system, gold)
            return tp, fp, tp, fn

    def docs_to_contingency(self, system, gold):
        return self.contingency([a for doc in system for a in doc.annotations],
                                [a for doc in gold for a in doc.annotations])
