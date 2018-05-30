# vim: set fileencoding=utf-8 :

from __future__ import division, print_function
from collections import defaultdict
import itertools
import operator
import random
import functools
import json

# Attempt to import joblib, but don't fail.
try:
    from joblib.parallel import Parallel, delayed, cpu_count
except ImportError:
    Parallel = delayed = cpu_count = None

#from data import measures, Reader
from .document import Reader
from .configs import (DEFAULT_MEASURE, parse_measures, MEASURE_HELP,
                      load_weighting)
from .evaluate import Evaluate, Matrix
from .utils import utf8_open


# 2500 bootstraps gives a robust CI lower-bound estimate to 3 significant
# figures on a TAC 2013 response under strong_link_measure
N_TRIALS = 2500


def json_format(self, data):
    return json.dumps(data, sort_keys=True, indent=4)


def no_format(self, data):
    return data


def sum(it, start=0):
    # Redefined to use __iadd__
    val = start  # XXX: copy?
    for o in it:
        val += o
    return val


def _result_diff(matrix1, matrix2):
    result1 = matrix1.results
    return {k: result1[k] - v
            for k, v in matrix2.results.items()}


def _permutation_trial(per_doc1, per_doc2):
    permuted = [(doc1, doc2) if random.random() > .5 else (doc2, doc1)
                for doc1, doc2 in zip(per_doc1, per_doc2)]
    pseudo1, pseudo2 = zip(*permuted)
    pseudo1 = sum(pseudo1, Matrix())
    pseudo2 = sum(pseudo2, Matrix())
    return _result_diff(pseudo1, pseudo2)


def count_permutation_trials(per_doc1, per_doc2, base_diff, n_trials):
    metrics, bases = zip(*base_diff.items())
    ops = [operator.le if base < 0 else operator.ge
           for base in bases]
    better = [0] * len(metrics)
    for _ in range(n_trials):
        result = _permutation_trial(per_doc1, per_doc2)
        for i, metric in enumerate(metrics):
            better[i] += ops[i](result[metric], bases[i])
    return dict(zip(metrics, better))


def _paired_bootstrap_trial(per_doc1, per_doc2):
    indices = [random.randint(0, len(per_doc1) - 1)
               for i in range(len(per_doc1))]
    pseudo1 = sum((per_doc1[i] for i in indices), Matrix())
    pseudo2 = sum((per_doc2[i] for i in indices), Matrix())
    return _result_diff(pseudo1, pseudo2)


def count_bootstrap_trials(per_doc1, per_doc2, base_diff, n_trials):
    # XXX: is this implementation correct?
    metrics, bases = zip(*base_diff.items())
    signs = [base >= 0 for base in bases]
    same_sign = [0] * len(metrics)
    for _ in range(n_trials):
        result = _paired_bootstrap_trial(per_doc1, per_doc2)
        for i, metric in enumerate(metrics):
            same_sign[i] += signs[i] == (result[metric] < 0)
    return dict(zip(metrics, same_sign))


def _job_shares(n_jobs, trials):
    if n_jobs == -1:
        n_jobs = cpu_count()
    shares = [trials // n_jobs] * n_jobs
    for i in range(trials - sum(shares)):
        shares[i] += 1
    return shares


class Significance(object):
    """Test for pairwise significance between systems"""

    METHODS = {'permute': count_permutation_trials,
               #'bootstrap': count_bootstrap_trials,
               }

    def __init__(self, systems, gold, trials=N_TRIALS, method='permute',
                 n_jobs=1, metrics=['precision', 'recall', 'fscore'],
                 fmt='none', measures=DEFAULT_MEASURE, type_weights=None):
        if len(systems) < 2:
            raise ValueError('Require at least two systems to compare')
        if method not in self.METHODS:
            raise ValueError('Unsupported method: {}'.format(method))
        # Check whether import worked, generate a more useful error.
        if Parallel is None:
            raise ImportError('Package: "joblib" not available, please install to run significance tests.')
        self.systems = systems
        self.gold = gold
        self.method = method
        self.trials = trials
        self.n_jobs = n_jobs
        self.measures = parse_measures(measures or DEFAULT_MEASURE,
                                       incl_clustering=False)
        self.metrics = metrics
        self.fmt = self.FMTS[fmt] if not callable(fmt) else fmt
        self.weighting = load_weighting(type_weights=type_weights)

    def __call__(self):
        all_counts = defaultdict(dict)
        #gold = sorted(Reader(utf8_open(self.gold)))
        gold = list(Reader(utf8_open(self.gold)))
        for path in self.systems:
            #system = sorted(Reader(utf8_open(path)))
            system = list(Reader(utf8_open(path)))
            doc_pairs = list(Evaluate.iter_pairs(system, gold))
            for measure, per_doc, overall in Evaluate.count_all(doc_pairs, self.measures, weighting=self.weighting):
                all_counts[measure][path] = (per_doc, overall)

        results = [{'sys1': sys1, 'sys2': sys2,
                    'measure': measure,
                    'stats': self.significance(measure_counts[sys1], measure_counts[sys2])}
                   for sys1, sys2 in itertools.combinations(self.systems, 2)
                   for measure, measure_counts in sorted(all_counts.items(),
                                                     key=lambda tup: self.measures.index(tup[0]))]

        return self.fmt(self, results)

    def significance(self, pair1, pair2):
        per_doc1, overall1 = pair1
        per_doc2, overall2 = pair2
        # TODO: limit to metrics
        base_diff = _result_diff(overall1, overall2)
        randomized_diffs = functools.partial(self.METHODS[self.method],
                                             per_doc1, per_doc2,
                                             base_diff)
        results = Parallel(n_jobs=self.n_jobs)(delayed(randomized_diffs)(share)
                                               for share in _job_shares(self.n_jobs, self.trials))
        all_counts = []
        for result in results:
            metrics, counts = zip(*result.items())
            all_counts.append(counts)

        return {metric: {'diff': base_diff[metric],
                         'p': (sum(counts) + 1) / (self.trials + 1)}
                for metric, counts in zip(metrics, zip(*all_counts))}

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('systems', nargs='+', metavar='FILE')
        p.add_argument('-g', '--gold', required=True)
        p.add_argument('-n', '--trials', default=N_TRIALS, type=int)
        p.add_argument('--permute', dest='method', action='store_const', const='permute',
                       default='permute',
                       help='Use the approximate randomization method')
        p.add_argument('--bootstrap', dest='method', action='store_const', const='bootstrap',
                       help='Use bootstrap resampling')
        p.add_argument('-j', '--n_jobs', default=1, type=int,
                       help='Number of parallel processes, use -1 for all CPUs')
        p.add_argument('-f', '--fmt', default='tab', choices=cls.FMTS.keys())
        p.add_argument('-m', '--measure', dest='measures', action='append',
                       metavar='NAME', help=MEASURE_HELP)
        p.add_argument('--type-weights', metavar='FILE', default=None,
                       help='File mapping gold and sys types to a weight, '
                       'such as produced by weights-for-hierarchy')
        p.add_argument('--metrics', default='precision recall fscore'.split(),
                       type=lambda x: x.split(','), help='Test significance for which metrics (default: precision,recall,fscore)')
        p.set_defaults(cls=cls)
        return p

    def tab_format(self, data):
        metrics = self.metrics
        rows = []
        for row in data:
            stats = row['stats']
            rows.append([row['sys1'], row['sys2'], row['measure']]
                        + sum(([stats[metric]['diff'], stats[metric]['p']]
                               for metric in metrics), []))
        header = (['sys1', 'sys2', 'measure'] +
                  sum(([u'Δ-' + metric[:6], 'p-' + metric[:6]]
                       for metric in metrics), []))

        sys_width = max(len(col) for row in rows for col in row[:2])
        sys_width = max(sys_width, 4)
        measure_width = max(len(row[2]) for row in rows)
        measure_width = max(measure_width, 5)

        fmt = (u'{:%ds}\t{:%ds}\t{:%ds}' % (sys_width, sys_width, measure_width))
        ret = (fmt + u'\t{}' * len(metrics) * 2).format(*header)
        fmt += u''.join(u'\t{:+8.3f}\t{:8.3f}' for metric in metrics)
        ret += u''.join(u'\n' + fmt.format(*row) for row in rows)
        return ret

    FMTS = {
        'tab': tab_format,
        'json': json_format,
        'none': no_format,
    }




def bootstrap_trials(per_doc, n_trials, metrics):
    """Bootstrap results over a single system output"""
    history = defaultdict(list)
    for _ in range(n_trials):
        indices = [random.randint(0, len(per_doc) - 1)
                   for i in range(len(per_doc))]
        result = sum((per_doc[i] for i in indices), Matrix()).results
        for metric in metrics:
            history[metric].append(result[metric])
    history.default_factory = None  # disable default
    return history


def _percentile(ordered, p):
    # As per http://www.itl.nist.gov/div898/handbook/prc/section2/prc252.htm
    k, d = divmod(p / 100 * (len(ordered) + 1), 1)
    # k is integer, d decimal part
    k = int(k)
    if 0 < k < len(ordered):
        lo, hi = ordered[k - 1:k + 1]
        return lo + d * (hi - lo)
    elif k == 0:
        return ordered[0]
    else:
        return ordered[-1]


class Confidence(object):
    """Calculate percentile bootstrap confidence intervals for a system
    """
    def __init__(self, system, gold, trials=N_TRIALS, percentiles=(90, 95, 99),
                 n_jobs=1, metrics=['precision', 'recall', 'fscore'],
                 measures=DEFAULT_MEASURE, fmt='none', type_weights=None):
        # Check whether import worked, generate a more useful error.
        if Parallel is None:
            raise ImportError('Package: "joblib" not available, please '
                              'install to run significance tests.')
        self.system = system
        self.gold = gold
        self.trials = trials
        self.n_jobs = n_jobs
        self.measures = parse_measures(measures or DEFAULT_MEASURE,
                                       incl_clustering=False)
        self.metrics = metrics
        self.percentiles = percentiles
        self.fmt = self.FMTS[fmt] if not callable(fmt) else fmt
        self.weighting = load_weighting(type_weights=type_weights)

    def calibrate_trials(self, trials=[100, 250, 500, 1000, 2500, 5000, 10000],
                         max_trials=20000):
        import numpy as np
        tmp_trials, self.trials = self.trials, max_trials
        matrices = self._read_to_matrices()
        print('measure', 'metric', 'pct', 'trials', 'stdev', sep='\t')
        for measure in self.measures:
            history = self.run_trials(matrices[measure][0])
            for metric in self.metrics:
                X = history[metric]
                for p in self.percentiles:
                    v = (100 - p) / 2
                    for n in trials:
                        stats = [_percentile(sorted(random.sample(X, n)), v)
                                 for i in range(100)]
                        print(measure, metric, p, n, np.std(stats), sep='\t')
        self.trials = tmp_trials

    def run_trials(self, per_doc):
        results = Parallel(n_jobs=self.n_jobs)(delayed(bootstrap_trials)(per_doc, share, self.metrics)
                                               for share in _job_shares(self.n_jobs, self.trials))
        history = defaultdict(list)
        for res in results:
            for metric in self.metrics:
                history[metric].extend(res[metric])
        return history

    def intervals(self, per_doc):
        history = self.run_trials(per_doc)
        ret = {}
        for metric, values in history.items():
            values.sort()
            ret[metric] = {p: (_percentile(values, (100 - p) / 2),
                               _percentile(values, 100 - (100 - p) / 2))
                           for p in self.percentiles}
        return ret

    def _read_to_matrices(self):
        gold = list(Reader(utf8_open(self.gold)))
        system = list(Reader(utf8_open(self.system)))
        doc_pairs = list(Evaluate.iter_pairs(system, gold))
        counts = {}
        for measure, per_doc, overall in Evaluate.count_all(doc_pairs, self.measures, weighting=self.weighting):
            counts[measure] = (per_doc, overall)
        return counts

    def calculate_all(self):
        counts = self._read_to_matrices()
        results = [{'measure': measure,
                    'overall': {k: v for k, v in overall.results.items() if k in self.metrics},
                    'intervals': self.intervals(per_doc)}
                   for measure, (per_doc, overall) in sorted(counts.items(),
                                                             key=lambda tup: self.measures.index(tup[0]))]
        return results

    def __call__(self):
        return self.fmt(self, self.calculate_all())

    def tab_format(self, data):
        # Input:
        # [{'measure': 'strong_mention_measure',
        #   'overall': {'precision': xx, 'recall': xx, 'fscore': xx},
        #   'intervals': {'precision': {90: [lo, hi]},
        #                 'recall': {90: [lo, hi]},
        #                 'fscore': {90: [lo, hi]}}},
        # ]
        percentiles = sorted(self.percentiles)
        header = ([u'measure', u'metric'] +
                  [u'{:d}%('.format(p) for p in reversed(percentiles)] +
                  [u'score'] +
                  [u'){:d}%'.format(p) for p in percentiles])

        # crazy formatting avoids lambda closure madness !
        meta_format = u'{{{{[intervals][{{metric}}][{}][{}]:.3f}}}}'
        formats = ([meta_format.format(p, 0) for p in reversed(percentiles)] +
                   [u'{{[overall][{metric}]:.3f}}'] +
                   [meta_format.format(p, 1) for p in percentiles])

        measure_width = max(map(len, self.measures))
        metric_width = max(map(len, self.metrics))
        fmt = (u'{:%ds}\t{:%ds}' % (measure_width, metric_width))
        rows = []
        for entry in data:
            for metric in self.metrics:
                rows.append([fmt.format(entry['measure'], metric)] +
                            [cell.format(metric=metric).format(entry)
                             for cell in formats])

        ret = (fmt + u'\t{}' * len(formats)).format(*header)
        ret += u''.join(u'\n' + u'\t'.join(row) for row in rows)
        return ret

    @staticmethod
    def read_tab_format(file):
        headers = [field.rstrip() for field in next(file).strip().split('\t')]
        by_measure = {}
        for line in file:
            row = dict(zip(headers, (field.rstrip() for field in line.rstrip().split('\t'))))
            measure = row['measure']
            if measure not in by_measure:
                cis = [int(field[:-2]) for field in headers
                       if field[-2:] == '%(']
                by_measure[measure] = {'measure': measure,
                                       'overall': {},
                                       'intervals': {metric: {} for metric in ('precision', 'recall', 'fscore')}}
            metric = row['metric']
            by_measure[measure]['overall'][metric] = float(row['score'])
            for ci in cis:
                by_measure[measure]['intervals'][metric][ci] = (float(row['%d%%(' % ci]), float(row[')%d%%' % ci]))
        return list(by_measure.values())

    FMTS = {'json': json_format,
            'tab': tab_format,
            'none': no_format}

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold', required=True)
        p.add_argument('-n', '--trials', default=N_TRIALS, type=int)
        p.add_argument('-j', '--n_jobs', default=1, type=int,
                       help='Number of parallel processes, use -1 for all CPUs')
        p.add_argument('-p', '--percentiles', default=(90, 95, 99),
                       type=lambda x: map(int, x.split(',')),
                       help='Output confidence intervals at these percentiles (default: 90,95,99)')
        p.add_argument('--metrics', default='precision recall fscore'.split(),
                       type=lambda x: x.split(','),
                       help='Calculate CIs for which metrics (default: precision,recall,fscore)')
        p.add_argument('-m', '--measure', dest='measures', action='append',
                       metavar='NAME', help=MEASURE_HELP)
        p.add_argument('--type-weights', metavar='FILE', default=None,
                       help='File mapping gold and sys types to a weight, '
                       'such as produced by weights-for-hierarchy')
        p.add_argument('-f', '--fmt', default='tab', choices=cls.FMTS.keys())
        p.set_defaults(cls=cls)
        return p
