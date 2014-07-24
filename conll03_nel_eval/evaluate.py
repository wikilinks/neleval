#!/usr/bin/env python
"""
Evaluate linker performance.
"""
from .coref_metrics import CMATCH_SETS, DEFAULT_CMATCH_SET, _to_matrix
from .document import Document, Reader
from .document import LMATCH_SETS, DEFAULT_LMATCH_SET
from .document import by_entity
from .utils import log
import json

METRICS = [
    'ptp',
    'fp',
    'rtp',
    'fn',
    'precision',
    'recall',
    'fscore',
]

FMTS = [
    'tab_format',
    'json_format',
    'no_format',
]
DEFAULT_FMT = 'tab_format'


class Evaluate(object):
    'Evaluate system output'

    def __init__(self, system, gold=None,
                 lmatches=DEFAULT_LMATCH_SET,
                 cmatches=DEFAULT_CMATCH_SET,
                 fmt=DEFAULT_FMT):
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
        self.format = getattr(self, fmt)
        if len(self.lmatches) > 0: # clust eval only
            self.doc_pairs = list(self.iter_pairs(self.system, self.gold))

    @classmethod
    def iter_pairs(self, system, gold):
        sdocs = {d.id:d for d in system}
        gdocs = {d.id:d for d in gold}
        for docid in set(sdocs.keys()).union(gdocs.keys()):
            sdoc = sdocs.get(docid) or Document(docid, [])
            gdoc = gdocs.get(docid) or Document(docid, [])
            yield sdoc, gdoc

    def __call__(self, lmatches=None, cmatches=None):
        lmatches = lmatches or self.lmatches
        self.results = self.link_eval(self.doc_pairs, lmatches)
        cmatches = cmatches or self.cmatches
        self.results.update(self.clust_eval(self.system, self.gold, cmatches))
        return self.format()

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.add_argument('-f', '--fmt', default=DEFAULT_FMT)
        p.add_argument('-l', '--lmatches', default=DEFAULT_LMATCH_SET,
                       choices=LMATCH_SETS.keys())
        p.add_argument('-c', '--cmatches', default=DEFAULT_CMATCH_SET,
                       choices=CMATCH_SETS.keys())
        p.set_defaults(cls=cls)
        return p

    @classmethod
    def link_eval(cls, doc_pairs, matches):
        results = {}
        for match, per_doc, overall in cls.count_all(doc_pairs, matches):
            results[match] = overall.results
        return results

    @classmethod
    def clust_eval(cls, system, gold, matches):
        results = {}
        sclust = dict(by_entity((a for d in system for a in d.annotations)))
        gclust = dict(by_entity((a for d in gold for a in d.annotations)))
        for m in matches:
            results[m.__name__] = Matrix.from_clust(sclust, gclust, m).results
        return results

    @classmethod
    def count_all(cls, doc_pairs, matches):
        for m in matches:
            yield (m,) + cls.count(m, doc_pairs)

    @classmethod
    def count(cls, match, doc_pairs):
        per_doc = []
        for sdoc, gdoc in doc_pairs:
            per_doc.append(Matrix.from_doc(sdoc, gdoc, match))
        overall = sum(per_doc, Matrix())
        return per_doc, overall

    # formatters

    def tab_format(self, num_fmt='{:.3f}', delimiter='\t'):
        lines = [delimiter.join([i[:6] for i in METRICS] + ['match'])]
        for lmatch in self.lmatches:
            row = self.row(lmatch, self.results, num_fmt)
            lines.append(delimiter.join(row))
        for cmatch in self.cmatches:
            row = self.row(cmatch.__name__, self.results, num_fmt)
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

    @classmethod
    def from_doc(cls, sdoc, gdoc, match):
        """
        Initialise from doc.
        sdoc - system Document object
        gdoc - gold Document object
        match - match method on doc
        """
        tp, fp, fn = getattr(gdoc, match)(sdoc)
        return cls(len(tp), len(fp), len(tp), len(fn))

    @classmethod
    def from_clust(cls, sclust, gclust, match):
        """
        Initialise from clustering.
        sclust - system clustering
        gclust - gold clustering
        match - coreference metric
        """
        # TODO remove?
        ptp, fp, rtp, fn = _to_matrix(*match(gclust, sclust))
        return cls(ptp, fp, rtp, fn)

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
            log.warn('Strict p/r defaulting to zero score for zero denominator')
            return 0.0
        else:
            return n / float(d)

    @property
    def fscore(self):
        p = self.precision
        r = self.recall
        return self.div(2*p*r, p+r)
