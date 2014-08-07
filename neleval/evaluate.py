#!/usr/bin/env python
"""
Evaluate linker performance.
"""
from .configs import DEFAULT_MATCH_SET, parse_matches, get_match_choices, get_matcher
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
                 matches=DEFAULT_MATCH_SET,
                 fmt='none'):
        """
        system - system output
        gold - gold standard
        matches - match definitions to use
        fmt - output format
        """
        self.system = list(Reader(open(system)))
        self.gold = list(Reader(open(gold)))
        self.matches = parse_matches(matches or DEFAULT_MATCH_SET)
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

    def __call__(self, matches=None):
        matches = parse_matches(matches) if matches is not None else self.matches
        self.results = {match: Matrix(*get_matcher(match).docs_to_contingency(self.system,
                                                                              self.gold)).results
                        for match in matches}
        return self.format(self)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.add_argument('-f', '--fmt', default='tab', choices=cls.FMTS.keys())
        p.add_argument('-m', '--matches', action='append', choices=get_match_choices())
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
            per_doc.append(Matrix(*matcher.contingency(sdoc.annotations,
                                                       gdoc.annotations)))
        overall = sum(per_doc, Matrix())
        return per_doc, overall

    # formatters

    def tab_format(self, num_fmt='{:.3f}', delimiter='\t'):
        lines = [delimiter.join([i[:6] for i in METRICS] + ['match'])]
        for match in self.matches:
            row = self.row(match, self.results, num_fmt)
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
