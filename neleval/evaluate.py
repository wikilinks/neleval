#!/usr/bin/env python
"""
Evaluate linker performance.
"""
from . import coref_metrics
from .configs import LMATCH_SETS, DEFAULT_LMATCH_SET, CMATCH_SETS, DEFAULT_CMATCH_SET
from .document import Document, Reader
import warnings
import json
from collections import Sequence, defaultdict
import operator


try:
    keys = dict.viewkeys
    import itertools
    filter = itertools.ifilter
except Exception:
    # Py3k
    keys = dict.keys


class StrictMetricWarning(Warning):
    pass


METRICS = [
    'ptp',
    'fp',
    'rtp',
    'fn',
    'precision',
    'recall',
    'fscore',
]


class Matcher(object):
    __slots__ = ['key', 'filter', 'agg']

    def __init__(self, key, filter=None, agg='micro'):
        """
        key : list of fields for mention comparison
        filter : a function or attribute name to select evaluated annotations
        agg : [work in progress]
        """
        if not isinstance(key, Sequence):
            raise TypeError('key should be a list or tuple')
        self.key = tuple(key)
        if filter is not None and not callable(filter):
            assert isinstance(filter, str)
            filter = operator.attrgetter(filter)
        self.filter = filter
        self.agg = agg

    NON_CLUSTERING_AGG = ('micro',)  # 'macro')

    def build_index(self, annotations):
        if isinstance(annotations, dict):
            # assume already built
            return annotations
        # TODO: caching

        key = self.key
        if self.filter is not None:
            annotations = filter(self.filter, annotations)
        return {tuple(getattr(ann, field) for field in key): ann
                for ann in annotations}

    def build_clusters(self, annotations):
        if isinstance(annotations, dict):
            # assume already built
            return annotations
        # TODO: caching

        key = self.key
        out = defaultdict(set)
        for ann in annotations:
            out[ann.eid].add(tuple(getattr(ann, field) for field in key))
        out.default_factory = None  # disable defaulting
        return out

    def count_matches(self, system, gold):
        if self.agg not in self.NON_CLUSTERING_AGG:
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
        if self.agg not in self.NON_CLUSTERING_AGG:
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
        if self.agg in self.NON_CLUSTERING_AGG:
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

    def to_matrix(self, system, gold):
        if self.agg in self.NON_CLUSTERING_AGG:
            tp, fp, fn = self.count_matches(system, gold)
            return Matrix(tp, fp, tp, fn)
        else:
            p_num, p_den, r_num, r_den = self.count_clustering(system, gold)
            ptp = p_num
            fp = p_den - p_num
            rtp = r_num
            fn = r_den - r_num
            return Matrix(ptp, fp, rtp, fn)

    def docs_to_matrix(self, system, gold):
        return self.to_matrix([a for doc in system for a in doc.annotations],
                              [a for doc in gold for a in doc.annotations])


MATCHERS = {
    'strong_mention_match':         Matcher(['span']),
    'strong_linked_mention_match':  Matcher(['span'], 'is_linked'),
    'strong_link_match':            Matcher(['span', 'kbid'], 'is_linked'),
    'strong_nil_match':             Matcher(['span'], 'is_nil'),
    'strong_all_match':             Matcher(['span', 'kbid']),
    'strong_typed_all_match':       Matcher(['span', 'type', 'kbid']),
    'entity_match':                 Matcher(['span', 'kbid'], 'is_linked'),

    'b_cubed_plus':                 Matcher(['span', 'kbid'], agg='b_cubed'),
}

for name in ['muc', 'b_cubed', 'entity_ceaf', 'mention_ceaf', 'pairwise',
             'cs_b_cubed', 'entity_cs_ceaf', 'mention_cs_ceaf']:
    MATCHERS[name] = Matcher(['span'], agg=name)


def get_matcher(name):
    if isinstance(name, Matcher):
        return name
    return MATCHERS[name]


class Evaluate(object):
    'Evaluate system output'

    def __init__(self, system, gold=None,
                 lmatches=DEFAULT_LMATCH_SET,
                 cmatches=DEFAULT_CMATCH_SET,
                 fmt='tab'):
        """
        system - system output
        gold - gold standard
        lmatches - link match definitions to use
        cmatches - cluster match definitions to use
        fmt - output format
        """
        self.system = list(Reader(open(system)))
        self.gold = list(Reader(open(gold)))
        self.lmatches = LMATCH_SETS[lmatches]
        self.cmatches = CMATCH_SETS[cmatches]
        self.format = self.FMTS[fmt] if fmt is not callable else fmt
        if len(self.lmatches) > 0:  # clust eval only
            self.doc_pairs = list(self.iter_pairs(self.system, self.gold))

    @classmethod
    def iter_pairs(self, system, gold):
        sdocs = {d.id: d for d in system}
        gdocs = {d.id: d for d in gold}
        for docid in set(sdocs.keys()).union(gdocs.keys()):
            sdoc = sdocs.get(docid) or Document(docid, [])
            gdoc = gdocs.get(docid) or Document(docid, [])
            yield sdoc, gdoc

    def __call__(self, lmatches=None, cmatches=None):
        matches = list(lmatches or self.lmatches) + list(cmatches or self.cmatches)
        self.results = {match: get_matcher(match).docs_to_matrix(self.system,
                                                                 self.gold).results
                        for match in matches}
        return self.format(self)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.add_argument('-f', '--fmt', default='tab', choices=cls.FMTS.keys())
        p.add_argument('-l', '--lmatches', default=DEFAULT_LMATCH_SET,
                       choices=LMATCH_SETS.keys())
        p.add_argument('-c', '--cmatches', default=DEFAULT_CMATCH_SET,
                       choices=CMATCH_SETS.keys())
        p.set_defaults(cls=cls)
        return p

    @classmethod
    def count_all(cls, doc_pairs, matches):
        for m in matches:
            yield (m,) + cls.count(m, doc_pairs)

    @classmethod
    def count(cls, match, doc_pairs):
        per_doc = []
        matcher = get_matcher(match)
        for sdoc, gdoc in doc_pairs:
            per_doc.append(matcher.to_matrix(sdoc.annotations,
                                             gdoc.annotations))
        overall = sum(per_doc, Matrix())
        return per_doc, overall

    # formatters

    def tab_format(self, num_fmt='{:.3f}', delimiter='\t'):
        lines = [delimiter.join([i[:6] for i in METRICS] + ['match'])]
        for lmatch in self.lmatches:
            row = self.row(lmatch, self.results, num_fmt)
            lines.append(delimiter.join(row))
        for cmatch in self.cmatches:
            row = self.row(cmatch, self.results, num_fmt)
            lines.append(delimiter.join(row))
        return '\n'.join(lines)

    def row(self, match_str, results, num_fmt):
        row = []
        match_results = self.results.get(match_str, {})
        for metric in METRICS:
            val = match_results.get(metric, 0)
            if isinstance(val, float):
                row.append(num_fmt.format(val))
            else:
                row.append(str(val))
        row.append(match_str)
        return row

    def json_format(self):
        return json.dumps(self.results, sort_keys=True, indent=4)

    def no_format(self):
        return self.results

    FMTS = {
        'tab': tab_format,
        'json': json_format,
        'none': no_format,
    }


class Matrix(object):
    def __init__(self, ptp=0, fp=0, rtp=0, fn=0):
        self.ptp = ptp
        self.fp = fp
        self.rtp = rtp
        self.fn = fn

    def __str__(self):
        return 'ptp={},fp={},rtp={},fn={}'.format(self.ptp,
                                                  self.fp,
                                                  self.rtp,
                                                  self.fn)

    def __add__(self, other):
        return Matrix(self.ptp + other.ptp,
                      self.fp + other.fp,
                      self.rtp + other.rtp,
                      self.fn + other.fn)

    def __iadd__(self, other):
        self.ptp += other.ptp
        self.fp += other.fp
        self.rtp += other.rtp
        self.fn += other.fn
        return self

    @property
    def results(self):
        return {
            'precision': self.precision,
            'recall': self.recall,
            'fscore': self.fscore,
            'ptp': self.ptp,
            'fp': self.fp,
            'rtp': self.rtp,
            'fn': self.fn,
            }

    @property
    def precision(self):
        return self.div(self.ptp, self.ptp+self.fp)

    @property
    def recall(self):
        return self.div(self.rtp, self.rtp+self.fn)

    def div(self, n, d):
        if d == 0:
            warnings.warn('Strict P/R defaulting to zero score for '
                          'zero denominator',
                          StrictMetricWarning)
            return 0.0
        else:
            return n / float(d)

    @property
    def fscore(self):
        p = self.precision
        r = self.recall
        return self.div(2*p*r, p+r)
