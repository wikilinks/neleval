#!/usr/bin/env python
"""
Evaluate linker performance.
"""
import json
from data import MATCHES, Reader
from utils import log

METRICS = [
    'tp',
    'fp',
    'fn',
    'precision',
    'recall',
    'fscore',
]

def tab_format(data, num_fmt='{:.3f}', delimiter='\t'):
    lines = [delimiter.join([i[:6] for i in METRICS] + ['match'])]
    for match in MATCHES:
        row = []
        for metric in METRICS:
            v = data.get(match, {}).get(metric, 0)
            if isinstance(v, float):
                row.append(num_fmt.format(v))
            else:
                row.append(str(v))
        row.append(match)
        lines.append(delimiter.join(row))
    return '\n'.join(lines)

def json_format(data):
    return json.dumps(data)

def no_format(data):
    return data

DEFAULT_FMT = 'tab'
FMTS = {
    'tab': tab_format,
    'json': json_format,
    'no_format': no_format,
}


class Evaluate(object):
    def __init__(self, system, gold=None, fmt=DEFAULT_FMT):
        """
        system - system output
        gold - gold standard
        fmt - format
        """
        self.system = sorted(Reader(open(system)))
        self.gold = sorted(Reader(open(system)))
        fmt_func = FMTS.get(fmt)
        assert fmt_func is not None
        self.fmt_func = fmt_func

    def __call__(self, matches=None):
        self.results = self.evaluate(self.system, self.gold, matches)
        return self.fmt_func(self.results)

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('evaluate', help='Evaluate system output')
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.add_argument('-f', '--fmt', default=DEFAULT_FMT)
        p.set_defaults(cls=cls)
        return p

    @classmethod
    def evaluate(cls, system, gold, matches=None):
        results = {}
        for match, per_doc, overall in cls.count_all(system, gold, matches):
            results[match] = overall.results
        return results

    @classmethod
    def count_all(cls, system, gold, matches=None):
        for m in matches or MATCHES:
            yield (m,) + cls.count(m, system, gold)

    @classmethod
    def count(cls, match, system, gold):
        per_doc = []
        for sdoc, gdoc in zip(system, gold):
            assert sdoc.doc_id == gdoc.doc_id, 'Require system and gold to be in the same order "{}" != "{}"'.format(sdoc.doc_id, gdoc.doc_id)
            per_doc.append(Matrix.from_doc(sdoc, gdoc, match))
        overall = sum(per_doc, Matrix())
        return per_doc, overall


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
    def from_doc(cls, sdoc, gdoc, match=MATCHES[0]):
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
