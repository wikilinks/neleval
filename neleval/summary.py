"""Tools to summarise the output of (multiple calls to) evaluation, confidence, etc."""

# TODO: PlotSystems: single figure for measures (in rows) and systems (by marker)
# TODO: PlotSystems: legend in plot
# TODO: examples for documentation (including different output formats)
# TODO: translation table for better display names for systems, measures, metrics
# TODO: custom figure size etc.
# TODO: scores as percentage?
from __future__ import print_function, absolute_import, division

import os
import itertools
import operator
import json
from collections import namedtuple
import re
import warnings

try:
    import numpy as np
except ImportError:
    np = None
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None
try:
    from scipy import stats
except ImportError:
    stats = None


from .configs import DEFAULT_MEASURE_SET, MEASURE_HELP, parse_measures
from .document import Reader
from .evaluate import Evaluate
from .significance import Confidence

DEFAULT_OUT_FMT = '.%s{}.pdf' % os.path.sep
MAX_LEGEND_PER_COL = 20


def _pairs(items):
    return itertools.combinations(items, 2)


def make_small_font():
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_size('small')
    return font


class _Result(namedtuple('Result', 'system measure data group')):
    def __new__(cls, system, measure, data, group=None):
        if group is None:
            group = system
        return super(_Result, cls).__new__(cls, system, measure, data, group)


XTICK_ROTATION = 45


class PlotSystems(object):
    """Summarise system results as scatter plots"""

    def __init__(self, systems, input_type='evaluate',
                 measures=DEFAULT_MEASURE_SET,
                 figures_by='measure', secondary='markers', prec_and_rec=False,
                 confidence=None, group_re=None, best_in_group=False,
                 out_fmt=DEFAULT_OUT_FMT, sort_by=None):
        if plt is None:
            raise ImportError('PlotSystems requires matplotlib to be installed')
        self.systems = systems
        self.measures = parse_measures(measures or DEFAULT_MEASURE_SET)
        self.input_type = input_type
        self.figures_by = figures_by or 'measure'
        self.confidence = confidence
        if confidence is not None and input_type != 'confidence':
            raise ValueError('--input-type=confidence required')
        self.secondary = secondary or 'markers'
        self.prec_and_rec = prec_and_rec

        self.out_fmt = out_fmt
        self.group_re = group_re
        self.best_in_group = best_in_group
        if self.best_in_group and \
           self.figures_by == 'measure' and \
           self.secondary == 'markers' and \
           len(self.measures) > 1:
            raise ValueError('best-in-group not supported with shared legend')
        if self.best_in_group and \
           self.figures_by == 'system' and \
           len(self.measures) > 1:
            raise ValueError('best-in-group cannot be evaluated with multiple measures per figure')

        self.sort_by = sort_by or 'none'
        if self.sort_by not in ('none', 'name', 'score') and \
           self.sort_by not in self.measures:
            raise ValueError('Acceptable values for sort-by are ("none", "name", "score", {})'.format(', '.join(map(repr, self.measures))))
        if self.sort_by == 'score' and \
           self.figures_by == 'system' and \
           len(self.measures) > 1:
            raise ValueError('Cannot sort by score with multiple measures per figure. You could instead specify a measure name.')

    def _plot(self, ax, x, y, *args, **kwargs):
        # uses errorbars where appropriate
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        fn = ax.scatter

        if x.dtype.names and 'lo' in x.dtype.names:
            kwargs['xerr'] = [x['score'] - x['lo'], x['hi'] - x['score']]
            kwargs['fmt'] = 'o'
            fn = ax.errorbar
        if x.dtype.names and 'score' in x.dtype.names:
            x = x['score']

        if y.dtype.names and 'lo' in y.dtype.names:
            kwargs['yerr'] = [y['score'] - y['lo'], y['hi'] - y['score']]
            kwargs['fmt'] = 'o'
            fn = ax.errorbar
        if y.dtype.names and 'score' in y.dtype.names:
            y = y['score']

        return fn(x, y, *args, **kwargs)

    def _plot1d(self, ax, all_scores, group_sizes):
        ordinate = np.repeat(np.arange(len(group_sizes)), group_sizes)
        if self.prec_and_rec:
            data = zip(['b', 'r'], ['precision', 'recall'], [all_scores[..., 0], all_scores[..., 1]])
            axis_label = 'precision/recall'
        else:
            data = zip(['k'], ['fscore'], [all_scores[..., 2]])
            axis_label = 'fscore'
        for color, label, scores in data:
            if self.secondary == 'rows':
                self._plot(ax, scores, ordinate[::-1], marker='.', color=color, label=label)
            else:
                self._plot(ax, ordinate, scores, marker='.', color=color, label=label)

        if self.secondary == 'rows':
            plt.xlabel(axis_label)
        else:
            plt.ylabel(axis_label)
        plt.legend()

    def _regroup(self, iterable, key, best_system=False, sort_by='name'):
        iterable = list(iterable)
        out = [(k, list(it)) for k, it in itertools.groupby(sorted(iterable, key=key), key=key)]
        if best_system:
            out = [(best.system, [best])
                   for best in (max(results, key=lambda result: result.data[2]['score']) for group, results in out)]
        if sort_by == 'name':
            # done above
            return out
        elif callable(sort_by):
            pass
        elif sort_by == 'measure':
            sort_by = lambda results: self.measures.index(results[0].measure)
        elif sort_by == 'score':
            sort_by = lambda results: -max(result.data[2]['score'] for result in results)
        else:
            raise ValueError('Unknown sort: {!r}'.format(sort_by))
        return sorted(out, key=lambda entry: sort_by(entry[1]))

    def _get_system_names(self, systems):
        path_prefix = os.path.commonprefix(systems)
        if os.path.sep in path_prefix:
            path_prefix = os.path.dirname(path_prefix) + os.path.sep
        path_suffix = os.path.commonprefix([system[::-1]
                                            for system in systems])
        return [(system[len(path_prefix):-len(path_suffix)],
                 self.group_re.search(system).group() if self.group_re else None)
                for system in systems]

    def __call__(self):
        # XXX: this needs a refactor/cleanup!!! Maybe just use more struct arrays rather than namedtuple
        if self.input_type == 'confidence':
            """
            {'intervals': {'fscore': {90: (0.504, 0.602),
                                      95: (0.494, 0.611),
                                      99: (0.474, 0.626)},
                            'precision': {90: (0.436, 0.56), 95: (0.426, 0.569), 99: (0.402, 0.591)},
                            'recall': {90: (0.573, 0.672), 95: (0.562, 0.681), 99: (0.543, 0.697)}},
             'measure': 'strong_nil_match',
             'overall': {'fscore': '0.555', 'precision': '0.498', 'recall': '0.626'}}
            """
            all_results = np.empty((len(self.systems), len(self.measures), 3), dtype=[('score', float),
                                                                                      ('lo', float),
                                                                                      ('hi', float)])
            for system, sys_results in zip(self.systems, all_results):
                result_dict = {entry['measure']: entry for entry in Confidence.read_tab_format(open(system))}
                # XXX: this is an ugly use of list comprehensions
                mat = [[(result_dict[measure]['overall'][metric], 0 if self.confidence is None else result_dict[measure]['intervals'][metric][self.confidence][0], 0 if self.confidence is None else result_dict[measure]['intervals'][metric][self.confidence][1])
                        for metric in ('precision', 'recall', 'fscore')]
                       for measure in self.measures]
                sys_results[...] = mat
            if self.confidence is None:
                # hide other fields
                all_results = all_results[['score']]

        else:
            all_results = np.empty((len(self.systems), len(self.measures), 3), dtype=[('score', float)])
            for system, sys_results in zip(self.systems, all_results):
                result_dict = Evaluate.read_tab_format(open(system))
                sys_results[...] = [[(result_dict[measure][metric],) for metric in ('precision', 'recall', 'fscore')]
                                    for measure in self.measures]

        # TODO: avoid legacy array intermediary
        all_results_tmp = []
        for (system_name, group), sys_results in zip(self._get_system_names(self.systems), all_results):
            all_results_tmp.extend(_Result(system=system_name, measure=measure, group=group, data=measure_results)
                                   for measure, measure_results in zip(self.measures, sys_results))
        all_results = all_results_tmp

        if self.sort_by in self.measures:
            by_measure = sorted((result for result in all_results if result.measure == self.sort_by), key=lambda result: -result.data[2]['score'])
            groups_by_measure = [result.group for result in by_measure]
            sort_by = lambda results: groups_by_measure.index(results[0].group)
        else:
            sort_by = self.sort_by

        if self.figures_by == 'measure':
            if sort_by == 'none':
                groups = [result.group for result in all_results]
                sort_by = lambda results: groups.index(results[0].group)
            primary_regroup = {'key': operator.attrgetter('measure')}
            secondary_regroup = {'key': operator.attrgetter('group'),
                                 'best_system': self.best_in_group,
                                 'sort_by': sort_by,}
        elif self.figures_by == 'system':
            if sort_by == 'none':
                sort_by = lambda results: self.measures.index(results[0].measure)
            primary_regroup = {'key': operator.attrgetter('group'),
                               'best_system': self.best_in_group}
            secondary_regroup = {'key': operator.attrgetter('measure'),
                                 'sort_by': sort_by,}
        else:
            raise ValueError('Unexpected figures_by: {!r}'.format(self.figures_by))
        get_primary = primary_regroup['key']
        get_secondary = secondary_regroup['key']

        n_secondary = len({get_secondary(res) for res in all_results})

        small_font = make_small_font()
        colors = plt.get_cmap('jet')(np.linspace(0, 1.0, n_secondary))
        for figure_name, figure_data in self._regroup(all_results, **primary_regroup):
            figure_data = self._regroup(figure_data, **secondary_regroup)  # TODO: sort
            markers = itertools.cycle(('+', '.', 'o', 's', '*', '^', 'v', 'p'))
            fig, ax = plt.subplots()
            if self.secondary == 'markers':
                patches = []
                for (secondary_name, results), color, marker in zip(figure_data, colors, markers):
                    # recall-precision
                    data = np.array([result.data for result in results])
                    patches.append(self._plot(ax, data[..., 1], data[..., 0],
                                              marker=marker, color=color,
                                              label=secondary_name))
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.axis((0, 1, 0, 1))
            else:
                secondary_names, figure_data = zip(*figure_data)

                ticks = np.arange(n_secondary)
                scores = np.array([result.data for results in figure_data for result in results])
                self._plot1d(ax, scores, [len(group) for group in figure_data])
                if self.secondary == 'rows':
                    plt.yticks(ticks[::-1], secondary_names, fontproperties=small_font)
                    plt.axis((0, 1, -.5, n_secondary - .5))
                elif self.secondary == 'columns':
                    plt.xticks(ticks, secondary_names, rotation=XTICK_ROTATION, fontproperties=small_font)
                    plt.axis((-.5, n_secondary - .5, 0, 1))
                else:
                    raise ValueError('Unexpected secondary: {!r}'.format(self.secondary))
            plt.tight_layout()  # would break axis resizing for markers layout

            # With CIs, non-score axis is clear enough
            plt.grid(axis='both' if self.confidence is None
                     else ('x' if self.secondary == 'rows' else 'y'))
            plt.savefig(self.out_fmt.format(figure_name))
            plt.close(fig)

        figure_names = sorted({get_primary(result) for result in all_results})

        if self.secondary == 'markers' and n_secondary > 1:
            # XXX: this uses `ax` defined above
            fig = plt.figure()
            legend = plt.figlegend(*ax.get_legend_handles_labels(), loc='center',
                                   ncol=int(np.ceil(n_secondary / MAX_LEGEND_PER_COL)),
                                   prop=small_font)
            fig.canvas.draw()
            # FIXME: need some padding
            bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(self.out_fmt.format('_legend_'), bbox_inches=bbox)
            figure_names.append('_legend_')

        return 'Saved to %s' % self.out_fmt.format('{%s}' % ','.join(figure_names))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('systems', nargs='+', metavar='FILE')
        meg = p.add_mutually_exclusive_group()
        meg.add_argument('--by-system', dest='figures_by', action='store_const', const='system',
                         help='Each system in its own figure')
        meg.add_argument('--by-measure', dest='figures_by', action='store_const', const='measure', default='measure',
                         help='Each measure in its own figure (default)')

        meg = p.add_mutually_exclusive_group()
        meg.add_argument('--2d', dest='secondary', action='store_const', const='markers', default='markers',
                         help='Plot precision and recall as separate axes with different markers as needed (default)')
        meg.add_argument('--rows', dest='secondary', action='store_const', const='rows',
                         help='Show rows of fscore plots')
        meg.add_argument('--columns', dest='secondary', action='store_const', const='columns',
                         help='Show columns of fscore plots')

        p.add_argument('--pr', dest='prec_and_rec', action='store_true', default=False,
                       help='In rows or columns mode, plot both precision and recall, rather than F1')

        p.add_argument('-i', '--input-type', choices=['evaluate', 'confidence'], default='evaluate',
                       help='Whether input was produced by the evaluate (default) or confidence command')
        p.add_argument('-o', '--out-fmt', default=DEFAULT_OUT_FMT,
                       help='Path template for saving plots with --fmt=plot (default: %(default)s))')
        p.add_argument('-m', '--measure', dest='measures', action='append',
                       metavar='NAME', help=MEASURE_HELP)

        p.add_argument('--ci', dest='confidence', type=int,
                       help='The percentile confidence interval to display as error bars '
                            '(requires --input-type=confidence')

        p.add_argument('--group-re', type=re.compile,
                       help='Display systems grouped, where a system\'s group label is extracted from its path by this PCRE')
        p.add_argument('--best-in-group', action='store_true', default=False,
                       help='Only show best system per group')
        p.add_argument('-s', '--sort-by',
                       help='Sort each plot, options include "none", "name", "score", or the name of a measure.')
        p.set_defaults(cls=cls)
        return p


class CompareMeasures(object):
    """Calculate statistics of measure distribution over systems
    """
    def __init__(self, systems, gold=None, evaluation_files=False,
                 measures=DEFAULT_MEASURE_SET,
                 fmt='none', out_fmt=DEFAULT_OUT_FMT,
                 sort_by='none'):
        if stats is None:
            raise ImportError('CompareMeasures requires scipy to be installed')
        self.systems = systems
        if gold:
            raise not evaluation_files
            self.gold = list(Reader(open(gold)))
        else:
            self.gold = None

        self.measures = parse_measures(measures or DEFAULT_MEASURE_SET)
        self.format = self.FMTS[fmt] if fmt is not callable else fmt
        self.out_fmt = out_fmt
        self.sort_by = sort_by

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
        small_font = make_small_font()
        correlations = results['correlations']

        measures = self.measures
        all_results = self.all_results

        # Order measures cleverly
        if self.sort_by == 'name':
            order = np.argsort(measures)
        elif self.sort_by == 'eigen':
            from matplotlib.mlab import PCA
            order = np.argsort(PCA(all_results).s)
        elif self.sort_by == 'mds':
            from sklearn.manifold import MDS
            order = np.argsort(MDS(n_components=1, n_init=20, random_state=0).fit_transform(all_results.T), axis=None)
        else:
            order = None
        if order is not None:
            measures = np.take(measures, order)
            all_results = np.take(all_results, order, axis=1)

        n_measures = len(measures)
        pearson = np.ma.masked_all((n_measures, n_measures), dtype=float)
        spearman = np.ma.masked_all((n_measures, n_measures), dtype=float)
        for (i, measure_i), (j, measure_j) in _pairs(enumerate(measures)):
            try:
                pair_corr = correlations[measure_i, measure_j]
            except KeyError:
                pair_corr = correlations[measure_j, measure_i]
            pearson[i, j] = pearson[j, i] = pair_corr['pearson'][0]
            spearman[i, j] = spearman[j, i] = pair_corr['spearman'][0]

        for i in range(n_measures):
            pearson[i, i] = spearman[i, i] = 1

        ticks = (np.arange(len(measures)), measures)

        cmap = cm.get_cmap('jet')  # or RdBu?
        cmap.set_bad('white')
        fig, ax = plt.subplots()
        im = ax.imshow(pearson, interpolation='nearest', cmap=cmap)
        plt.colorbar(im)
        plt.xticks(*ticks, rotation=XTICK_ROTATION, fontproperties=small_font)
        plt.yticks(*ticks, fontproperties=small_font)
        plt.tight_layout()
        plt.savefig(self.out_fmt.format('pearson'))
        plt.close(fig)

        fig, ax = plt.subplots()
        im = ax.imshow(spearman, interpolation='nearest', cmap=cmap)
        plt.colorbar(im)
        plt.xticks(*ticks, rotation=XTICK_ROTATION, fontproperties=small_font)
        plt.yticks(*ticks, fontproperties=small_font)
        plt.tight_layout()
        plt.savefig(self.out_fmt.format('spearman'))

        fig, ax = plt.subplots()
        ax.boxplot(all_results[:, ::-1], 0, 'rs', 0, labels=measures[::-1])
        plt.yticks(fontproperties=small_font)
        plt.tight_layout()
        plt.savefig(self.out_fmt.format('spread'))

        return 'Saved to %s' % self.out_fmt.format('{pearson,spearman,spread}')


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
        meg.add_argument('-e', '--evaluation-files', action='store_true', default=False,
                         help='System paths are the tab-formatted outputs '
                              'of the evaluate command, rather than '
                              'system outputs')
        p.add_argument('-f', '--fmt', default='tab', choices=cls.FMTS.keys())
        p.add_argument('-o', '--out-fmt', default=DEFAULT_OUT_FMT,
                       help='Path template for saving plots with --fmt=plot (default: %(default)s))')
        p.add_argument('-m', '--measure', dest='measures', action='append',
                       metavar='NAME', help=MEASURE_HELP)
        p.add_argument('-s', '--sort-by', choices=['none', 'name', 'eigen', 'mds'],
                       help='For plot, sort by name, eigenvalue, or '
                            'multidimensional scaling (requires scikit-learn)')
        p.set_defaults(cls=cls)
        return p
