#!/usr/bin/env python
"""
Evaluate linker performance.
"""
from .document import Document, MATCH_SETS, DEFAULT_MATCH_SET, Reader
import json

METRICS = [
    'tp',
    'fp',
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
DEFAULT_FMT = 'tab'


class Evaluate(object):
    'Evaluate system output'

    def __init__(self, system, gold=None, matches=DEFAULT_MATCH_SET,
                 fmt=DEFAULT_FMT):
        """
        system - system output
        gold - gold standard
        matches - match definitions to use
        fmt - format
        """
        self.system = Reader(open(system))
        self.gold = Reader(open(gold))
        self.matches = MATCH_SETS[matches]
        self.format = getattr(self, fmt)
        self.doc_pairs = list(self.iter_pairs())

    def iter_pairs(self):
        sdocs = {d.id:d for d in self.system}
        gdocs = {d.id:d for d in self.gold}
        for docid in set(sdocs.keys()).union(gdocs.keys()):
            sdoc = sdocs.get(docid) or Document(docid, [])
            gdoc = gdocs.get(docid) or Document(docid, [])
            yield sdoc, gdoc

    def __call__(self, matches=None):
        self.results = self.evaluate(self.doc_pairs, matches or self.matches)
        return self.format()

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.add_argument('-f', '--fmt', default=DEFAULT_FMT)
        p.add_argument('-m', '--matches', default='hachey_acl14',
                       choices=MATCH_SETS.keys())
        p.set_defaults(cls=cls)
        return p

    @classmethod
    def evaluate(cls, doc_pairs, matches):
        results = {}
        for match, per_doc, overall in cls.count_all(doc_pairs, matches):
            results[match] = overall.results
        return results

    @classmethod
    def count_all(cls, doc_pairs, matches):
        for m in matches:
            yield (m,) + cls.count(m, doc_pairs)

    @classmethod
    def count(cls, match, doc_pairs):
        per_doc = []
        for sdoc, gdoc in doc_pairs:
            print sdoc
            print gdoc
            per_doc.append(Matrix.from_doc(sdoc, gdoc, match))
        overall = sum(per_doc, Matrix())
        return per_doc, overall

    # formatters

    def tab_format(self, num_fmt='{:.3f}', delimiter='\t'):
        lines = [delimiter.join([i[:6] for i in METRICS] + ['match'])]
        for match in self.matches:
            row = []
            for metric in METRICS:
                v = self.results.get(match, {}).get(metric, 0)
                if isinstance(v, float):
                    row.append(num_fmt.format(v))
                else:
                    row.append(str(v))
            row.append(match)
            lines.append(delimiter.join(row))
        return '\n'.join(lines)

    def json_format(self):
        return json.dumps(self.results)

    def no_format(self):
        return self.results


class Matrix(object):
    def __init__(self, tp=0, fp=0, fn=0):
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __str__(self):
        return 'tp={},fp={},fn={}'.format(self.tp, self.fp, self.fn)

    def __add__(self, other):
        return Matrix(self.tp + other.tp,
                      self.fp + other.fp,
                      self.fn + other.fn)

    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
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
        return cls(len(tp), len(fp), len(fn))

    @property
    def results(self):
        return {
            'precision': self.precision,
            'recall': self.recall,
            'fscore': self.fscore,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            }

    @property
    def precision(self):
        return self.div(self.tp, self.tp+self.fp)

    @property
    def recall(self):
        return self.div(self.tp, self.tp+self.fn)

    def div(self, n, d):
        return 1.0 if d == 0 else n / float(d)

    @property
    def fscore(self):
        p = self.precision
        r = self.recall
        return self.div(2*p*r, p+r)
