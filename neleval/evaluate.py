#!/usr/bin/env python
"""
Evaluate linker performance.
"""
from .configs import (DEFAULT_MEASURE_SET, parse_measures,
                      MEASURE_HELP, get_measure)
from .document import Document, Reader
import warnings
import json


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


class Evaluate(object):
    'Evaluate system output'

    def __init__(self, system, gold=None,
                 measures=DEFAULT_MEASURE_SET,
                 fmt='none'):
        """
        system - system output
        gold - gold standard
        measures - measure definitions to use
        fmt - output format
        """
        if not isinstance(system, list):
            system = list(Reader(open(system)))
        if not isinstance(gold, list):
            gold = list(Reader(open(gold)))
        self.system = system
        self.gold = gold
        self.measures = parse_measures(measures or DEFAULT_MEASURE_SET)
        self.format = self.FMTS[fmt] if fmt is not callable else fmt
        self.doc_pairs = list(self.iter_pairs(self.system, self.gold))

    @classmethod
    def iter_pairs(self, system, gold):
        sdocs = {d.id: d for d in system}
        gdocs = {d.id: d for d in gold}
        for docid in set(sdocs.keys()).union(gdocs.keys()):
            sdoc = sdocs.get(docid) or Document(docid, [])
            gdoc = gdocs.get(docid) or Document(docid, [])
            yield sdoc, gdoc

    def __call__(self, measures=None):
        measures = (parse_measures(measures)
                    if measures is not None
                    else self.measures)
        self.results = {measure: Matrix(*get_measure(measure).
                                        docs_to_contingency(self.system,
                                                            self.gold)).results
                        for measure in measures}
        return self.format(self, self.results)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold', required=True)
        p.add_argument('-f', '--fmt', default='tab', choices=cls.FMTS.keys())
        p.add_argument('-m', '--measure', dest='measures', action='append',
                       metavar='NAME', help=MEASURE_HELP)
        p.set_defaults(cls=cls)
        return p

    @classmethod
    def count_all(cls, doc_pairs, measures):
        for m in measures:
            yield (m,) + cls.count(m, doc_pairs)

    @classmethod
    def count(cls, measure, doc_pairs):
        per_doc = []
        measure = get_measure(measure)
        for sdoc, gdoc in doc_pairs:
            per_doc.append(Matrix(*measure.contingency(sdoc.annotations,
                                                       gdoc.annotations)))
        overall = sum(per_doc, Matrix())
        return per_doc, overall

    # formatters

    def tab_format(self, results, num_fmt='{:.3f}', delimiter='\t'):
        lines = [self._header(delimiter)]
        for measure in self.measures:
            row = self.row(results, measure, num_fmt)
            lines.append(delimiter.join(row))
        return '\n'.join(lines)

    @staticmethod
    def _header(delimiter='\t'):
        return delimiter.join([i[:6] for i in METRICS] + ['measure'])

    def row(self, results, measure_str, num_fmt):
        row = []
        measure_results = results.get(measure_str, {})
        for metric in METRICS:
            val = measure_results.get(metric, 0)
            if isinstance(val, float):
                row.append(num_fmt.format(val))
            else:
                row.append(str(val))
        row.append(measure_str)
        return row

    @classmethod
    def read_tab_format(cls, file):
        header = next(file)
        assert header.rstrip() == cls._header()
        results = {}
        for l in file:
            row = l.rstrip().split('\t')
            measure = row.pop()
            row = map(float, row)
            results[measure] = dict(zip(METRICS, row))
        return results

    def json_format(self, results):
        return json.dumps(results, sort_keys=True, indent=4)

    def no_format(self, results):
        return results

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
