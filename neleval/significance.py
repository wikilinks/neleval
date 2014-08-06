# vim: set fileencoding=utf-8 :

from __future__ import division, print_function
from collections import defaultdict
import itertools
import random
import operator
import functools
import json

# Attempt to import joblib, but don't fail.
try:
    from joblib.parallel import Parallel, delayed, cpu_count
except ImportError:
    Parallel = delayed = cpu_count = None

#from data import MATCHES, Reader
from document import Reader, LMATCH_SETS, DEFAULT_LMATCH_SET
from evaluate import Evaluate, Matrix


# 2500 bootstraps gives a robust CI lower-bound estimate to 3 significant
# figures on a TAC 2013 response under strong_link_match
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
            for k, v in matrix2.results.iteritems()}


def _permutation_trial(per_doc1, per_doc2):
    permuted = [(doc1, doc2) if random.random() > .5 else (doc2, doc1)
                for doc1, doc2 in zip(per_doc1, per_doc2)]
    pseudo1, pseudo2 = zip(*permuted)
    pseudo1 = sum(pseudo1, Matrix())
    pseudo2 = sum(pseudo2, Matrix())
    return _result_diff(pseudo1, pseudo2)


def count_permutation_trials(per_doc1, per_doc2, base_diff, n_trials):
    metrics, bases = zip(*base_diff.iteritems())
    ops = [operator.le if base < 0 else operator.ge
           for base in bases]
    better = [0] * len(metrics)
    for _ in xrange(n_trials):
        result = _permutation_trial(per_doc1, per_doc2)
        for i, metric in enumerate(metrics):
            better[i] += ops[i](result[metric], bases[i])
    return dict(zip(metrics, better))


def _paired_bootstrap_trial(per_doc1, per_doc2):
    indices = [random.randint(0, len(per_doc1) - 1)
               for i in xrange(len(per_doc1))]
    pseudo1 = sum((per_doc1[i] for i in indices), Matrix())
    pseudo2 = sum((per_doc2[i] for i in indices), Matrix())
    return _result_diff(pseudo1, pseudo2)


def count_bootstrap_trials(per_doc1, per_doc2, base_diff, n_trials):
    # XXX: is this implementation correct?
    metrics, bases = zip(*base_diff.iteritems())
    signs = [base >= 0 for base in bases]
    same_sign = [0] * len(metrics)
    for _ in xrange(n_trials):
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
                 fmt='tab', lmatches=DEFAULT_LMATCH_SET):
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
        self.lmatches = LMATCH_SETS[lmatches]
        self.metrics = metrics
        self.fmt = self.FMTS[fmt] if fmt is not callable else fmt

    def __call__(self):
        all_counts = defaultdict(dict)
        #gold = sorted(Reader(open(self.gold)))
        gold = list(Reader(open(self.gold)))
        for path in self.systems:
            #system = sorted(Reader(open(path)))
            system = list(Reader(open(path)))
            doc_pairs = list(Evaluate.iter_pairs(system, gold))
            for match, per_doc, overall in Evaluate.count_all(doc_pairs, self.lmatches):
                all_counts[match][path] = (per_doc, overall)

        results = [{'sys1': sys1, 'sys2': sys2,
                    'match': match,
                    'stats': self.significance(match_counts[sys1], match_counts[sys2])}
                   for sys1, sys2 in itertools.combinations(self.systems, 2)
                   for match, match_counts in sorted(all_counts.iteritems(),
                                                     key=lambda (k, v): self.lmatches.index(k))]

        return self.fmt(self, results)

    def significance(self, (per_doc1, overall1), (per_doc2, overall2)):
        # TODO: limit to metrics
        base_diff = _result_diff(overall1, overall2)
        randomized_diffs = functools.partial(self.METHODS[self.method],
                                             per_doc1, per_doc2,
                                             base_diff)
        results = Parallel(n_jobs=self.n_jobs)(delayed(randomized_diffs)(share)
                                               for share in _job_shares(self.n_jobs, self.trials))
        all_counts = []
        for result in results:
            metrics, counts = zip(*result.iteritems())
            all_counts.append(counts)

        return {metric: {'diff': base_diff[metric],
                         'p': (sum(counts) + 1) / (self.trials + 1)}
                for metric, counts in zip(metrics, zip(*all_counts))}

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('systems', nargs='+', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.add_argument('-n', '--trials', default=N_TRIALS, type=int)
        p.add_argument('--permute', dest='method', action='store_const', const='permute',
                       default='permute',
                       help='Use the approximate randomization method')
        p.add_argument('--bootstrap', dest='method', action='store_const', const='bootstrap',
                       help='Use bootstrap resampling')
        p.add_argument('-j', '--n_jobs', default=1, type=int,
                       help='Number of parallel processes, use -1 for all CPUs')
        p.add_argument('-f', '--fmt', default='tab', choices=cls.FMTS.keys())
        p.add_argument('--metrics', default='precision recall fscore'.split(),
                       type=lambda x: x.split(','), help='Test significance for which metrics (default: precision,recall,fscore)')
        p.add_argument('-l', '--lmatches', default=DEFAULT_LMATCH_SET,
                       choices=LMATCH_SETS.keys())
        p.set_defaults(cls=cls)
        return p

    def tab_format(self, data):
        metrics = self.metrics
        rows = []
        for row in data:
            stats = row['stats']
            rows.append([row['sys1'], row['sys2'], row['match']]
                        + sum(([stats[metric]['diff'], stats[metric]['p']]
                               for metric in metrics), []))
        header = (['sys1', 'sys2', 'match'] +
                  sum(([u'Δ-' + metric[:6], 'p-' + metric[:6]]
                       for metric in metrics), []))

        sys_width = max(len(col) for row in rows for col in row[:2])
        sys_width = max(sys_width, 4)
        match_width = max(len(row[2]) for row in rows)
        match_width = max(match_width, 5)

        fmt = (u'{:%ds}\t{:%ds}\t{:%ds}' % (sys_width, sys_width, match_width))
        ret = (fmt + u'\t{}' * len(metrics) * 2).format(*header)
        fmt += u''.join(u'\t{:+8.3f}\t{:8.3f}' for metric in metrics)
        ret += u''.join(u'\n' + fmt.format(*row) for row in rows)
        return ret.encode('utf-8')

    FMTS = {
        'tab': tab_format,
        'json': json_format,
        'none': no_format,
    }




def bootstrap_trials(per_doc, n_trials, metrics):
    """Bootstrap results over a single system output"""
    history = defaultdict(list)
    for _ in xrange(n_trials):
        indices = [random.randint(0, len(per_doc) - 1)
                   for i in xrange(len(per_doc))]
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
                 lmatches=DEFAULT_LMATCH_SET, fmt='tab'):
        # Check whether import worked, generate a more useful error.
        if Parallel is None:
            raise ImportError('Package: "joblib" not available, please '
                              'install to run significance tests.')
        self.system = system
        self.gold = gold
        self.trials = trials
        self.n_jobs = n_jobs
        self.lmatches = LMATCH_SETS.get(lmatches, lmatches)
        self.metrics = metrics
        self.percentiles = percentiles
        self.fmt = self.FMTS[fmt] if fmt is not callable else fmt

    def calibrate_trials(self, trials=[100, 250, 500, 1000, 2500, 5000, 10000],
                         max_trials=20000):
        import numpy as np
        tmp_trials, self.trials = self.trials, max_trials
        matrices = self._read_to_matrices()
        print('match', 'metric', 'pct', 'trials', 'stdev', sep='\t')
        for match in self.lmatches:
            history = self.run_trials(matrices[match][0])
            for metric in self.metrics:
                X = history[metric]
                for p in self.percentiles:
                    v = (100 - p) / 2
                    for n in trials:
                        stats = [_percentile(sorted(random.sample(X, n)), v)
                                 for i in range(100)]
                        print(match, metric, p, n, np.std(stats), sep='\t')
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
        gold = list(Reader(open(self.gold)))
        system = list(Reader(open(self.system)))
        doc_pairs = list(Evaluate.iter_pairs(system, gold))
        counts = {}
        for match, per_doc, overall in Evaluate.count_all(doc_pairs, self.lmatches):
            counts[match] = (per_doc, overall)
        return counts

    def calculate_all(self):
        counts = self._read_to_matrices()
        results = [{'match': match,
                    'overall': {k: v for k, v in overall.results.items() if k in self.metrics},
                    'intervals': self.intervals(per_doc)}
                   for match, (per_doc, overall) in sorted(counts.iteritems(),
                                                           key=lambda (k, v): self.lmatches.index(k))]
        return results

    def __call__(self):
        return self.fmt(self, self.calculate_all())

    def tab_format(self, data):
        # Input:
        # [{'match': 'strong_mention_match',
        #   'overall': {'precision': xx, 'recall': xx, 'fscore': xx},
        #   'intervals': {'precision': {90: [lo, hi]},
        #                 'recall': {90: [lo, hi]},
        #                 'fscore': {90: [lo, hi]}}},
        # ]
        percentiles = sorted(self.percentiles)
        header = ([u'match', u'metric'] +
                  [u'{:d}%('.format(p) for p in percentiles] +
                  [u'score'] +
                  [u'){:d}%'.format(p) for p in reversed(percentiles)])

        # crazy formatting avoids lambda closure madness !
        meta_format = u'{{{{[intervals][{{metric}}][{}][{}]:.3f}}}}'
        formats = ([meta_format.format(p, 0) for p in percentiles] +
                   [u'{{[overall][{metric}]:.3f}}'] +
                   [meta_format.format(p, 1) for p in reversed(percentiles)])

        lmatch_width = max(map(len, self.lmatches))
        metric_width = max(map(len, self.metrics))
        fmt = (u'{:%ds}\t{:%ds}' % (lmatch_width, metric_width))
        rows = []
        for entry in data:
            for metric in self.metrics:
                rows.append([fmt.format(entry['match'], metric)] +
                            [cell.format(metric=metric).format(entry)
                             for cell in formats])

        ret = (fmt + u'\t{}' * len(formats)).format(*header)
        ret += u''.join(u'\n' + u'\t'.join(row) for row in rows)
        return ret.encode('utf-8')

    FMTS = {'json': json_format,
            'tab': tab_format,
            'none': no_format}

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.add_argument('-n', '--trials', default=N_TRIALS, type=int)
        p.add_argument('-j', '--n_jobs', default=1, type=int,
                       help='Number of parallel processes, use -1 for all CPUs')
        p.add_argument('--percentiles', default=(90, 95, 99),
                       type=lambda x: map(int, x.split(',')),
                       help='Output confidence intervals at these percentiles (default: 90,95,99)')
        p.add_argument('--metrics', default='precision recall fscore'.split(),
                       type=lambda x: x.split(','),
                       help='Calculate CIs for which metrics (default: precision,recall,fscore)')
        p.add_argument('-l', '--lmatches', default=DEFAULT_LMATCH_SET,
                       choices=LMATCH_SETS.keys())
        p.add_argument('-f', '--fmt', default='tab', choices=cls.FMTS.keys())
        p.set_defaults(cls=cls)
        return p
