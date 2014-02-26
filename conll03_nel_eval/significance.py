from __future__ import division
from collections import defaultdict
import itertools
import random
import operator
import functools

from joblib.parallel import Parallel, delayed, cpu_count

from data import Reader
from evaluate import Evaluate, Matrix


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


def _bootstrap_trial(per_doc1, per_doc2):
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
        result = _bootstrap_trial(per_doc1, per_doc2)
        for i, metric in enumerate(metrics):
            same_sign[i] += signs[i] == (result[metric] >= 0)
    return dict(zip(metrics, same_sign))


class Significance(object):
    """Test for pairwise significance between systems"""

    METHODS = {'permute': count_permutation_trials,
               #'bootstrap': count_bootstrap_trials,
               }

    def __init__(self, systems, gold, trials=10000, method='permute',
                 n_jobs=1):
        if len(systems) < 2:
            raise ValueError('Require at least two systems to compare')
        if method not in self.METHODS:
            raise ValueError('Unsupported method: {}'.format(method))
        self.systems = systems
        self.gold = gold
        self.method = method
        self.trials = trials
        self.n_jobs = n_jobs

    def __call__(self):
        all_counts = defaultdict(dict)
        gold = sorted(Reader(open(self.gold)))
        for path in self.systems:
            system = sorted(Reader(open(path)))
            for match, per_doc, overall in Evaluate.count_all(system, gold):
                all_counts[match][path] = (per_doc, overall)

        results = [(sys1, sys2, match,
                    self.significance(match_counts[sys1], match_counts[sys2]))
                   for sys1, sys2 in itertools.combinations(self.systems, 2)
                   for match, match_counts in all_counts.iteritems()]

        # TODO: format results
        return results

    def significance(self, (per_doc1, overall1), (per_doc2, overall2)):
        base_diff = _result_diff(overall1, overall2)
        randomized_diffs = functools.partial(self.METHODS[self.method],
                                             per_doc1, per_doc2,
                                             base_diff)
        n_jobs = self.n_jobs
        if n_jobs == -1:
            n_jobs = cpu_count()
        shares = [self.trials // n_jobs] * n_jobs
        for i in range(self.trials - sum(shares)):
            shares[i] += 1

        results = Parallel(n_jobs=self.n_jobs)(delayed(randomized_diffs)(share)
                                               for share in shares)
        all_counts = []
        for result in results:
            metrics, counts = zip(*result.iteritems())
            all_counts.append(counts)

        return {metric: (sum(counts) + 1) / (self.trials + 1)
                for metric, counts in zip(metrics, zip(*all_counts))}

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('significance', help=cls.__doc__)
        p.add_argument('systems', nargs='+', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.add_argument('-n', '--trials', default=10000, type=int)
        p.add_argument('--permute', dest='method', action='store_const', const='permute',
                       default='permute',
                       help='Use the approximate randomization method')
        p.add_argument('--bootstrap', dest='method', action='store_const', const='bootstrap',
                       help='Use bootstrap resampling')
        p.add_argument('-j', '--n_jobs', default=1, type=int,
                       help='Number of parallel processes, use -1 for all CPUs')
        p.set_defaults(cls=cls)
        return p
