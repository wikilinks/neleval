#!/usr/bin/env python
"""
Evaluate linker performance.
"""
import pprint
from data import Data

MATCH = 'strong_link_match'

class Evaluate(object):
    def __init__(self, fname, gold, match):
        """
        fname - system output
        gold - gold standard
        match - mention match method to use
        """
        self.match = match
	pprint.pprint(self.evaluate(fname, gold))

    def evaluate(self, system, gold):
        self.system = Data.from_file(system)
        self.gold = Data.from_file(gold)
        self.tp, self.fp, self.fn = self.load()
        return self.results

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('evaluate', help='Evaluate system output')
        p.add_argument('-g', '--gold')
        p.add_argument('-m', '--match', default=MATCH)
        p.set_defaults(cls=cls)
        return p

    def load(self):
        """
        Calculate tp, fp, fn per doc.
        """
        tp = {} # {doc_id: tp_count}
        fp = {} # {doc_id: fp_count}
        fn = {} # {doc_id: fn_count}
        for id, (sdoc, gdoc) in self._docs:
            sg_tp, sg_fp = self._load(sdoc, gdoc)
            gs_tp, gs_fp = self._load(gdoc, sdoc)
            assert sg_tp == gs_tp
            tp[id] = sg_tp
            fp[id] = sg_fp
            fn[id] = gs_fp
        return tp, fp, fn

    def _load(self, sdoc, gdoc):
        tp = 0
        fp = 0
        for smen in sdoc.mentions:
            matches = []
            for gmen in gdoc.mentions:
                if getattr(smen, self.match)(gmen):
                    matches.append(smen)
            if len(matches) == 0:
                fp += 1
            else:
                tp += 1
        return tp, fp

    @property
    def results(self):
        return {
            'micro_precision': self.micro_precision,
            'micro_recall': self.micro_recall,
            'micro_fscore': self.micro_fscore,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'macro_fscore': self.macro_fscore,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            }

    @property
    def micro_precision(self):
        tp = 0.0
        fp = 0.0
        for id, _ in self._docs:
            tp += self.tp[id]
            fp += self.fp[id]
        return self._precision(tp, fp)

    @property
    def macro_precision(self):
        p = 0.0
        n = 0
        for id, _ in self._docs:
            p += self._precision(self.tp[id], self.fp[id])
            n += 1
        return p / n

    def _precision(self, tp, fp):
        if (tp+fp) == 0:
            return 1
        else:
            return tp / float(tp+fp)

    @property
    def micro_recall(self):
        tp = 0.0
        fn = 0.0
        for id, _ in self._docs:
            tp += self.tp[id]
            fn += self.fn[id]
        return self._recall(tp, fn)

    @property
    def macro_recall(self):
        r = 0.0
        n = 0
        for id, _ in self._docs:
            r += self._recall(self.tp[id], self.fn[id])
            n += 1
        return r / n

    def _recall(self, tp, fn):
        if (tp+fn) == 0:
            return 1
        else:
            return tp / float(tp+fn)

    @property
    def micro_fscore(self):
        return self._harmonic_mean(self.micro_precision, self.micro_recall)

    @property
    def macro_fscore(self):
        return self._harmonic_mean(self.macro_precision, self.macro_recall)

    def _harmonic_mean(self, p, r):
        return 2*p*r / float(p+r)

    @property
    def _docs(self):
        for id, sdoc in self.system.documents.iteritems():
            if sdoc is None:
                continue
            gdoc = self.gold.documents[id]
            if gdoc is None:
                continue
            yield id, (sdoc, gdoc)
