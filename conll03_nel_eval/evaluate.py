#!/usr/bin/env python
"""
Evaluate linker performance.
"""
import pprint
from data import Data

MATCH = 'strong_link_match'

class Evaluate(object):
    def __init__(self, fname, gold, match=MATCH):
        """
        fname - system output
        gold - gold standard
        match - match method
        """
        self.match = match
        self.evaluate(fname, gold)
        pprint.pprint(self.accumulated.results)

    def evaluate(self, system, gold):
        self.system = Data.from_file(system)
        self.gold = Data.from_file(gold)
        self.matrixes, self.accumulated = self.load()

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('evaluate', help='Evaluate system output')
        p.add_argument('-g', '--gold')
        p.add_argument('-m', '--match', default=MATCH)
        p.set_defaults(cls=cls)
        return p

    def load(self):
        matrixes = [] # doc-level matrixes
        accumulator = Matrix(0, 0, 0) # accumulator matrix
        for sdoc, gdoc in self._docs:
            m = Matrix.from_doc(sdoc, gdoc, self.match)
            matrixes.append(m)
            accumulator = accumulator + m
        return matrixes, accumulator

    @property
    def _docs(self):
        for id, sdoc in self.system.documents.iteritems():
            if sdoc is None:
                continue
            gdoc = self.gold.documents[id]
            if gdoc is None:
                continue
            yield sdoc, gdoc

class Matrix(object):
    def __init__(self, tp, fp, fn):
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __add__(self, other):
        return Matrix(self.tp + other.tp,
                      self.fp + other.fp,
                      self.fn + other.fn)

    @classmethod
    def from_doc(cls, sdoc, gdoc, match=MATCH):
        """
        Initialise from doc.
        sdoc - system Document object
        gdoc - gold Document object
        match - match method on doc
        """
        sg_tp, fp = getattr(sdoc, match)(gdoc)
        gs_tp, fn = getattr(gdoc, match)(sdoc)
        assert sg_tp == gs_tp
        return cls(sg_tp, fp, fn)

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
        return 2*p*r / float(p+r)
