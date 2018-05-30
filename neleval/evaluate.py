#!/usr/bin/env python
"""
Evaluate linker performance.
"""
import warnings
import json
import itertools
from collections import OrderedDict, defaultdict

from .configs import (DEFAULT_MEASURE_SET, parse_measures,
                      MEASURE_HELP, get_measure, load_weighting)
from .document import Document, Reader
from .utils import log, utf8_open, json_dumps


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
                 fmt='none', group_by=None, overall=False,
                 type_weights=None):
        """
        system - system output
        gold - gold standard
        measures - measure definitions to use
        fmt - output format
        """
        if not isinstance(system, list):
            log.debug('Reading system output..')
            system = list(Reader(utf8_open(system)))
            log.debug('..done.')
        if not isinstance(gold, list):
            log.debug('Reading gold standard..')
            gold = list(Reader(utf8_open(gold)))
            log.debug('..done.')
        self.system = system
        self.gold = gold
        self.measures = parse_measures(measures or DEFAULT_MEASURE_SET)
        self.format = self.FMTS[fmt] if not callable(fmt) else fmt
        self.doc_pairs = list(self.iter_pairs(self.system, self.gold))
        self.group_by = group_by
        self.overall = overall
        self.weighting = load_weighting(type_weights=type_weights)

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
        self.results = OrderedDict()

        # XXX: should avoid grouping by doc in the first place
        system_annotations = [ann for doc in self.system
                              for ann in doc.annotations]
        gold_annotations = [ann for doc in self.gold
                            for ann in doc.annotations]
        if not self.group_by:
            name_fmt = '{measure}'
            system_grouped = {((),): system_annotations}
            gold_grouped = {((),): gold_annotations}
        else:
            name_fmt = '{measure}'
            for i, field in enumerate(self.group_by):
                name_fmt += ';{}={{group[{}]}}'.format(field, i)
            get_group = lambda ann: tuple(getattr(ann, field)
                                          for field in self.group_by)

            system_grouped = defaultdict(list)
            for ann in system_annotations:
                system_grouped[get_group(ann)].append(ann)

            gold_grouped = defaultdict(list)
            for ann in gold_annotations:
                gold_grouped[get_group(ann)].append(ann)

        if self.group_by:
            n_fields = len(self.group_by)
            group_vals = [sorted(set(group[i] for group in gold_grouped))
                          for i in range(n_fields)]
        else:
            group_vals = [((),)]

        for measure in measures:
            measure_mats = []
            for group in itertools.product(*group_vals):
                # XXX should we only be accounting for groups that are non-empty in gold? non empty in either sys or gold?
                mat = Matrix(*get_measure(measure, weighting=self.weighting).
                             contingency(system_grouped[group],
                                         gold_grouped[group])
                             )
                measure_mats.append((group, mat))

                if not self.group_by or not self.overall:
                    name = name_fmt.format(measure=measure,
                                           group=[json.dumps(v) for v in group])
                    self.results[name] = mat.results

            if self.group_by:
                # Macro-averages for each field
                micro_labels = ['<micro>'] * len(self.group_by)
                for i in range(n_fields):
                    constituents = defaultdict(Matrix)
                    for group, mat in measure_mats:
                        constituents[group[i]] += mat

                    labels = micro_labels[:]
                    labels[i] = '<macro>'
                    name = name_fmt.format(measure=measure, group=labels)
                    self.results[name] = macro_average(constituents.values())

                # Overall micro-average
                name = name_fmt.format(measure=measure, group=micro_labels)
                self.results[name] = sum(constituents.values(),
                                         Matrix()).results

        return self.format(self, self.results)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold', required=True)
        p.add_argument('-f', '--fmt', default='tab', choices=cls.FMTS.keys())
        p.add_argument('-m', '--measure', dest='measures', action='append',
                       metavar='NAME', help=MEASURE_HELP)
        p.add_argument('-b', '--group-by', dest='group_by', action='append',
                       metavar='FIELD',
                       help='Report results per field-value, '
                            'and micro/macro-averaged over these, '
                            'Multiple --group-by may be used.  '
                            'E.g. -b docid -b type.  '
                            'NB: micro-average may not equal overall score.')
        p.add_argument('--by-doc', dest='group_by', action='append_const',
                       const='docid', help='Alias for -b docid')
        p.add_argument('--by-type', dest='group_by', action='append_const',
                       const='type', help='Alias for -b type')
        p.add_argument('--overall', default=False, action='store_true',
                       help='With --group-by, report only overall, not per-group results')
        p.add_argument('--type-weights', metavar='FILE', default=None,
                       help='File mapping gold and sys types to a weight, '
                       'such as produced by weights-for-hierarchy')
        p.set_defaults(cls=cls)
        return p

    @classmethod
    def count_all(cls, doc_pairs, measures, weighting=None):
        for m in measures:
            yield (m,) + cls.count(m, doc_pairs, weighting=weighting)

    @classmethod
    def count(cls, measure, doc_pairs, weighting=None):
        per_doc = []
        measure = get_measure(measure, weighting=weighting)
        for sdoc, gdoc in doc_pairs:
            per_doc.append(Matrix(*measure.contingency(sdoc.annotations,
                                                       gdoc.annotations)))
        overall = sum(per_doc, Matrix())
        return per_doc, overall

    # formatters

    def tab_format(self, results, num_fmt='{:.3f}', delimiter='\t'):
        lines = [self._header(delimiter)]
        for measure, measure_results in sorted(results.items()):
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
        assert header.rstrip() == cls._header(), 'Differing headers: expected {!r}, got {!r}'.format(cls._header(), header.rstrip())
        results = {}
        for l in file:
            row = l.rstrip().split('\t')
            measure = row.pop()
            row = map(float, row)
            results[measure] = dict(zip(METRICS, row))
        return results

    def json_format(self, results):
        return json_dumps(results)

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


def macro_average(results_iter):
    out = defaultdict(float)
    for i, results in enumerate(results_iter):
        if hasattr(results, 'results'):
            # accept Matrix instances instead
            results = results.results
        for k, v in results.items():
            out[k] += v
    return {k: v / (i + 1) for k, v in out.items()}
