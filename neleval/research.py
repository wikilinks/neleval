"""
Tools related to researching NEL evaluation
"""
from .configs import DEFAULT_MEASURE_SET, MEASURE_HELP, parse_measures
from .document import Reader
from .evaluate import Evaluate

import itertools
import json
import os

try:
    from scipy import stats
    import numpy as np
except ImportError:
    stats = None

DEFAULT_OUT_PREFIX = '.' + os.path.sep

def _pairs(items):
    return itertools.combinations(items, 2)


class CompareMeasures(object):
    """Calculate statistics of measure distribution over systems
    """
    def __init__(self, systems, gold=None, evaluation_files=False,
                 measures=DEFAULT_MEASURE_SET,
                 fmt='none', out_prefix=DEFAULT_OUT_PREFIX,
                 sort_measures=True):
        """
        system - system output
        gold - gold standard
        measures - measure definitions to use
        fmt - output format
        """
        if stats is None:
            raise ImportError('CompareMeasures requires scipy to be installed')
        self.systems = systems
        if gold:
            raise not evaluation_files
            self.gold = list(Reader(open(gold)))
        else:
            self.gold = None

        self.measures = parse_measures(measures or DEFAULT_MEASURE_SET)
        if sort_measures:
            self.measures.sort()
        self.format = self.FMTS[fmt] if fmt is not callable else fmt
        self.out_prefix = out_prefix

    def __call__(self):
        all_results = np.empty((len(self.systems), len(self.measures)))
        # TODO: parallelise?
        for system, sys_results in zip(self.systems, all_results):
            if self.gold is None:
                result_dict = Evaluate.read_tab_format(open(system))
            else:
                result_dict = Evaluate(system, self.gold, measures=self.measures, fmt='none')()
            sys_results[...] = [result_dict[measure]['fscore'] for measure in self.measures]
        self.all_results = all_results
        correlations = {}
        scores_by_measure = zip(self.measures, all_results.T)
        for (measure_i, scores_i), (measure_j, scores_j) in _pairs(scores_by_measure):
            correlations[measure_i, measure_j] = {'pearson': stats.pearsonr(scores_i, scores_j),
                                                  'spearman': stats.spearmanr(scores_i, scores_j)}

        quartiles = {}
        for measure_i, scores_i in scores_by_measure:
            quartiles[measure_i] = np.percentile(scores_i, [0, 25, 50, 75, 100])

        return self.format(self, {'quartiles': quartiles, 'correlations': correlations})

    def tab_format(self, results):
        correlations = results['correlations']
        quartiles = results['quartiles']
        rows = [['measure1', 'measure2', 'pearson-r', 'spearman-r', 'median-diff', 'iqr-ratio']]
        for measure1, measure2 in _pairs(self.measures):
            pair_corr = correlations[measure1, measure2]
            quart1 = quartiles[measure1]
            quart2 = quartiles[measure2]
            data = [pair_corr['pearson'][0], pair_corr['spearman'][0],
                    quart1[2] - quart2[2],
                    (quart1[3] - quart1[1]) / (quart2[3] - quart2[1])]
            data = ['%0.3f' % v for v in data]
            rows.append([measure1, measure2] + data)

        col_widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
        fmt = '\t'.join('{{:{:d}s}}'.format(width) for width in col_widths)
        return "\n".join(fmt.format(*row) for row in rows)

    def json_format(self, results):
        return json.dumps(results, sort_keys=True, indent=4)

    def no_format(self, results):
        return results

    def plot_format(self, results):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        correlations = results['correlations']
        n_measures = len(self.measures)
        pearson = np.ma.masked_all((n_measures, n_measures), dtype=float)
        spearman = np.ma.masked_all((n_measures, n_measures), dtype=float)
        for (i, measure_i), (j, measure_j) in _pairs(enumerate(self.measures)):
            cell = (i, j)
            pearson[cell] = correlations[measure_i, measure_j]['pearson'][0]
            spearman[cell] = correlations[measure_i, measure_j]['spearman'][0]

        ticks = (np.arange(len(self.measures)), self.measures)

        cmap = cm.get_cmap('jet')  # or RdBu?
        cmap.set_bad('white')
        fig, ax = plt.subplots()
        im = ax.imshow(pearson, interpolation='nearest', cmap=cmap)
        plt.colorbar(im)
        plt.xticks(*ticks, rotation='vertical')
        plt.yticks(*ticks)
        plt.tight_layout()
        plt.savefig(self.out_prefix + 'pearson.pdf')

        fig, ax = plt.subplots()
        im = ax.imshow(spearman, interpolation='nearest', cmap=cmap)
        plt.colorbar(im)
        plt.xticks(*ticks, rotation='vertical')
        plt.yticks(*ticks)
        plt.tight_layout()
        plt.savefig(self.out_prefix + 'spearman.pdf')

        fig, ax = plt.subplots()
        ax.boxplot(self.all_results[:, ::-1], 0, 'rs', 0, labels=self.measures[::-1])
        plt.tight_layout()
        plt.savefig(self.out_prefix + 'spread.pdf')

        return 'Saved to %s{pearson,spearman,spread}.pdf' % self.out_prefix


    FMTS = {
        'none': no_format,
        'tab': tab_format,
        'json': json_format,
        'plot': plot_format,
    }

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('systems', nargs='+', metavar='FILE')
        meg = p.add_mutually_exclusive_group(required=True)
        meg.add_argument('-g', '--gold')
        meg.add_argument('-e', '--evaluation-files',
                         help='System paths are the tab-formatted outputs '
                              'of the evaluate command, rather than '
                              'system outputs')
        p.add_argument('-f', '--fmt', default='tab', choices=cls.FMTS.keys())
        p.add_argument('-o', '--out-prefix', default=DEFAULT_OUT_PREFIX,
                       help='Prefix for saving plots with --fmt=plot (default: %(default)s))')
        p.add_argument('-m', '--measure', dest='measures', action='append',
                       metavar='NAME', help=MEASURE_HELP)
        p.add_argument('--no-sort', dest='sort_measures', default=True,
                       action='store_false',
                       help='Do not sort measures alphabetically')
        p.set_defaults(cls=cls)
        return p
