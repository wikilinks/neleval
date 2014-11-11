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
from .interact import embed_shell

DEFAULT_OUT_FMT = '.%s{}.pdf' % os.path.sep
MAX_LEGEND_PER_COL = 20
CMAP = 'jet'


def _pairs(items):
    return itertools.combinations(items, 2)


def make_small_font():
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_size('small')
    return font


def _parse_figsize(figsize):
    if figsize.count(',') != 1:
        raise ValueError('Expected a single comma in figure size, got {!r}'.format(figsize))
    width, _, height = figsize.partition(',')
    return int(width), int(height)


class _Result(namedtuple('Result', 'system measure data group')):
    def __new__(cls, system, measure, data, group=None):
        if group is None:
            group = system
        return super(_Result, cls).__new__(cls, system, measure, data, group)


XTICK_ROTATION = 'vertical'


class PlotSystems(object):
    """Summarise system results as scatter plots"""

    def __init__(self, systems, input_type='evaluate',
                 measures=DEFAULT_MEASURE_SET,
                 figures_by='measure', secondary='markers', metrics=('fscore',),
                 lines=False,
                 confidence=None, group_re=None, best_in_group=False,
                 sort_by=None,
                 out_fmt=DEFAULT_OUT_FMT, figsize=(8, 6), interactive=False):
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
        self.metrics = metrics

        self.lines = lines
        self.interactive = interactive
        self.out_fmt = out_fmt
        self.figsize = figsize

        self.group_re = group_re
        self.best_in_group = best_in_group
        if self.best_in_group and \
           self.figures_by == 'measure' and \
           self.secondary == 'markers' and \
           len(self.measures) > 1:
            raise ValueError('best-in-group not supported with shared legend')
        self.sort_by = sort_by or 'none'
        if self.sort_by not in ('none', 'name', 'score') and \
           self.sort_by not in self.measures:
            raise ValueError('Acceptable values for sort-by are ("none", "name", "score", {})'.format(', '.join(map(repr, self.measures))))

        multiple_measures_per_figure = (secondary == 'heatmap') or (self.figures_by == 'system' and len(self.measures) > 1)
        if self.best_in_group and multiple_measures_per_figure:
            raise ValueError('best-in-group cannot be evaluated with multiple measures per figure')
        if self.sort_by == 'score' and multiple_measures_per_figure:
            raise ValueError('Cannot sort by score with multiple measures per figure. You could instead specify a measure name.')

    def _plot(self, ax, x, y, *args, **kwargs):
        # uses errorbars where appropriate
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        fn = ax.plot

        if x.dtype.names and 'lo' in x.dtype.names:
            kwargs['xerr'] = [x['score'] - x['lo'], x['hi'] - x['score']]
            fn = ax.errorbar
        if x.dtype.names and 'score' in x.dtype.names:
            x = x['score']

        if y.dtype.names and 'lo' in y.dtype.names:
            kwargs['yerr'] = [y['score'] - y['lo'], y['hi'] - y['score']]
            fn = ax.errorbar
        if y.dtype.names and 'score' in y.dtype.names:
            y = y['score']

        if fn == ax.plot:
            kwargs['ls'] = '-' if self.lines else 'None'
        else:
            kwargs['fmt'] = '-o' if self.lines else 'o'

        return fn(x, y, *args, **kwargs)

    METRIC_DATA = {'precision': ('b', 0, '^'), 'recall': ('r', 1, 'v'), 'fscore': ('k', 2, 's')}

    def _plot1d(self, ax, all_scores, group_sizes):
        ordinate = np.repeat(np.arange(len(group_sizes)), group_sizes)
        # TODO: clean this up: use kwargs in METRIC_DATA
        colors, columns, markers = zip(*(self.METRIC_DATA[metric] for metric in self.metrics))
        data = zip(colors, self.metrics, [all_scores[..., c] for c in columns], markers)
        if tuple(self.metrics) == ('fscore',):
            axis_label = 'fscore'
        else:
            axis_label = 'score'
        for color, label, scores, marker in data:
            if self.secondary == 'rows':
                self._plot(ax, scores, ordinate[::-1], marker=marker, color=color, label=label, markeredgecolor=color)
            else:
                self._plot(ax, ordinate, scores, marker=marker, color=color, label=label, markeredgecolor=color)

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

    def _load_data(self):
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
        return all_results_tmp

    def __call__(self):
        all_results = self._load_data()

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
            primary_regroup = {'key': operator.attrgetter('measure'),
                               'sort_by': 'measure',}
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

        if self.interactive:
            figures = {}
        else:
            figure_names = []
        for name, figure, save_kwargs in self._generate_figures(all_results, primary_regroup, secondary_regroup):
            if self.interactive:
                figures[name] = figure
            else:
                figure_names.append(name)
                figure.savefig(self.out_fmt.format(name), **save_kwargs)
                plt.close(figure)

        if self.interactive:
            print('Opening interactive shell with variables `figures` and `results`')
            embed_shell({'figures': figures, 'results': all_results})
        else:
            return 'Saved to %s' % self.out_fmt.format('{%s}' % ','.join(figure_names))

    def _generate_figures(self, *args):
        if self.secondary == 'heatmap':
            yield self._heatmap(*args)
        else:
            for plot in self._generate_plots(*args):
                yield plot

    def _heatmap(self, all_results, primary_regroup, secondary_regroup):
        # FIXME: sort_by only currently applied to columns!
        figure = plt.figure('heatmap', figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)

        matrix = []
        row_names = []
        for row_name, row in self._regroup(all_results, **primary_regroup):
            column_names, row = zip(*self._regroup(row, **secondary_regroup))
            matrix.append([cell.data[2]['score'] for (cell,) in row])
            row_names.append(row_name)

        im = ax.imshow(matrix, interpolation='nearest',
                       cmap=plt.get_cmap(CMAP), vmin=0, vmax=1)
        small_font = make_small_font()
        plt.yticks(np.arange(len(row_names)), row_names,
                   fontproperties=small_font)
        plt.xticks(np.arange(len(column_names)), column_names,
                   rotation=XTICK_ROTATION, fontproperties=small_font)
        figure.colorbar(im)
        figure.tight_layout()
        return 'heatmap', figure, {}

    def _marker_cycle(self):
        return itertools.cycle(('+', '.', 'o', 's', '*', '^', 'v', 'p'))

    def _generate_plots(self, all_results, primary_regroup, secondary_regroup):
        small_font = make_small_font()
        for figure_name, figure_data in self._regroup(all_results, **primary_regroup):
            figure_data = self._regroup(figure_data, **secondary_regroup)
            n_secondary = len(figure_data)
            colors = plt.get_cmap(CMAP)(np.linspace(0, 1.0, n_secondary))
            fig = plt.figure(figure_name, figsize=self.figsize)
            ax = fig.add_subplot(1, 1, 1)
            if self.secondary == 'markers':
                markers = self._marker_cycle()
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
            fig.tight_layout()  # would break axis resizing for markers layout

            plt.grid(axis='x' if self.secondary == 'rows' else 'y')
            yield figure_name, fig, {}

        if self.secondary == 'markers' and n_secondary > 1:
            # XXX: this uses `ax` defined above
            fig = plt.figure()
            legend = plt.figlegend(*ax.get_axes().get_legend_handles_labels(), loc='center',
                                   ncol=int(np.ceil(n_secondary / MAX_LEGEND_PER_COL)),
                                   prop=make_small_font())
            fig.canvas.draw()
            # FIXME: need some padding
            bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            yield '_legend_', fig, {'bbox_inches': bbox}

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('systems', nargs='+', metavar='FILE')
        meg = p.add_mutually_exclusive_group()
        meg.add_argument('--by-system', dest='figures_by', action='store_const', const='system',
                         help='Each system in its own figure, or row with --heatmap')
        meg.add_argument('--by-measure', dest='figures_by', action='store_const', const='measure', default='measure',
                         help='Each measure in its own figure, or row with --heatmap (default)')

        meg = p.add_mutually_exclusive_group()
        meg.add_argument('--scatter', dest='secondary', action='store_const', const='markers', default='markers',
                         help='Plot precision and recall as separate axes with different markers as needed (default)')
        meg.add_argument('--rows', dest='secondary', action='store_const', const='rows',
                         help='Show rows of fscore plots')
        meg.add_argument('--columns', dest='secondary', action='store_const', const='columns',
                         help='Show columns of fscore plots')
        meg.add_argument('--heatmap', dest='secondary', action='store_const', const='heatmap',
                         help='Show a heatmap comparing all systems and measures')

        meg = p.add_mutually_exclusive_group()
        meg.add_argument('--pr', dest='metrics', action='store_const', const=('precision', 'recall'), default=('fscore',),
                         help='In rows or columns mode, plot both precision and recall, rather than F1')
        meg.add_argument('--prf', dest='metrics', action='store_const', const=('precision', 'recall', 'fscore'),
                         help='In rows or columns mode, plot precision and recall as well as F1')

        p.add_argument('--lines', action='store_true', default=False,
                       help='Draw lines between points in rows/cols mode')

        p.add_argument('-i', '--input-type', choices=['evaluate', 'confidence'], default='evaluate',
                       help='Whether input was produced by the evaluate (default) or confidence command')

        meg = p.add_mutually_exclusive_group()
        meg.add_argument('-o', '--out-fmt', default=DEFAULT_OUT_FMT,
                       help='Path template for saving plots with --fmt=plot (default: %(default)s))')
        meg.add_argument('--interactive', action='store_true', default=False,
                         help='Open an interactive shell with `figures` available instead of saving images to file')

        p.add_argument('--figsize', default=(8, 6), type=_parse_figsize,
                       help='The width,height of a figure in inches (default 8,6)')

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
                 fmt='none', out_fmt=DEFAULT_OUT_FMT, figsize=(8, 6),
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
        self.figsize = figsize
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

        cmap = plt.get_cmap(CMAP)
        cmap.set_bad('white')
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(pearson, interpolation='nearest', cmap=cmap)
        plt.colorbar(im)
        plt.xticks(*ticks, rotation=XTICK_ROTATION, fontproperties=small_font)
        plt.yticks(*ticks, fontproperties=small_font)
        plt.tight_layout()
        plt.savefig(self.out_fmt.format('pearson'))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(spearman, interpolation='nearest', cmap=cmap)
        plt.colorbar(im)
        plt.xticks(*ticks, rotation=XTICK_ROTATION, fontproperties=small_font)
        plt.yticks(*ticks, fontproperties=small_font)
        plt.tight_layout()
        plt.savefig(self.out_fmt.format('spearman'))

        fig, ax = plt.subplots(figsize=self.figsize)
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
        p.add_argument('--figsize', default=(8, 6), type=_parse_figsize,
                       help='The width,height of a figure in inches (default 8,6)')

        p.add_argument('-m', '--measure', dest='measures', action='append',
                       metavar='NAME', help=MEASURE_HELP)
        p.add_argument('-s', '--sort-by', choices=['none', 'name', 'eigen', 'mds'],
                       help='For plot, sort by name, eigenvalue, or '
                            'multidimensional scaling (requires scikit-learn)')
        p.set_defaults(cls=cls)
        return p
