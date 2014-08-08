import textwrap
from collections import defaultdict
from .annotation import Measure

try:
    keys = dict.viewkeys
except Exception:
    # Py3k
    keys = dict.keys


MEASURES = {
    'strong_mention_match':         Measure(['span']),
    'strong_linked_mention_match':  Measure(['span'], 'is_linked'),
    'strong_link_match':            Measure(['span', 'kbid'], 'is_linked'),
    'strong_nil_match':             Measure(['span'], 'is_nil'),
    'strong_all_match':             Measure(['span', 'kbid']),
    'strong_typed_all_match':       Measure(['span', 'type', 'kbid']),
    'entity_match':                 Measure(['docid', 'kbid'], 'is_linked'),

    'b_cubed_plus':                 Measure(['span', 'kbid'], agg='b_cubed'),
}

for name in ['muc', 'b_cubed', 'entity_ceaf', 'mention_ceaf', 'pairwise',
             #'cs_b_cubed', 'entity_cs_ceaf', 'mention_cs_ceaf']:
             ]:
    MEASURES[name] = Measure(['span'], agg=name)


# Configuration constants
ALL_MEASURES = 'all'
ALL_TAGGING = 'all-tagging'
ALL_COREF = 'all-coref'
TAC_MEASURES = 'tac'
TAC14_MEASURES = 'tac14'
TMP_MEASURES = 'tmp'
CORNOLTI_WWW13_MEASURES = 'cornolti'
HACHEY_ACL14_MEASURES = 'hachey'
LUO_MEASURES = 'luo'
CAI_STRUBE_MEASURES = 'cai'

MEASURE_SETS = {
    ALL_MEASURES: [
        'all-tagging',
        'all-coref',
        ],
    ALL_TAGGING: {
        'strong_mention_match',
        'strong_linked_mention_match',
        'strong_link_match',
        'strong_nil_match',
        'strong_all_match',
        'strong_typed_all_match',
        'entity_match',
    },
    ALL_COREF: {
        'mention_ceaf',
        'entity_ceaf',
        'b_cubed',
        'pairwise',
        'muc',
        #'mention_cs_ceaf',
        #'entity_cs_ceaf',
        #'cs_b_cubed',
        'b_cubed_plus',
    },
    CORNOLTI_WWW13_MEASURES: [
        'strong_linked_mention_match',
        'strong_link_match',
        'entity_match',
        ],
    HACHEY_ACL14_MEASURES: [
        'strong_mention_match',  # full ner
        'strong_linked_mention_match',
        'strong_link_match',
        'entity_match',
        ],
    LUO_MEASURES: [
        'muc',
        'b_cubed',
        'mention_ceaf',
        'entity_ceaf',
        ],
    #CAI_STRUBE_MEASURES: [
    #    'cs_b_cubed',
    #    'entity_cs_ceaf',
    #    'mention_cs_ceaf',
    #],
    TAC_MEASURES: [
        'strong_link_match',  # recall equivalent to kb accuracy before 2014
        'strong_nil_match',  # recall equivalent to nil accuracy before 2014
        'strong_all_match',  # equivalent to overall accuracy before 2014
        'strong_typed_all_match',  # wikification f-score for TAC 2014

        'mention_ceaf',
        'b_cubed',
        'b_cubed_plus',
        ],
    TAC14_MEASURES: [
        'strong_typed_all_match',  # wikification f-score for TAC 2014
    ],
    TMP_MEASURES: [
        'mention_ceaf',
        'entity_ceaf',
        'pairwise',
        ],
}

DEFAULT_MEASURE_SET = ALL_MEASURES
DEFAULT_MEASURE = 'strong_all_match'


def _expand(measures):
    if isinstance(measures, str):
        if measures in MEASURE_SETS:
            measures = MEASURE_SETS[measures]
        else:
            return [measures]
    if isinstance(measures, Measure):
        return [measures]
    if len(measures) == 1:
        return _expand(measures[0])
    return [m for group in measures for m in _expand(group)]


def parse_measures(in_measures, incl_clustering=True):
    # flatten nested sequences and expand group names
    measures = _expand(in_measures)
    # remove duplicates while maintaining order
    seen = set()
    measures = [seen.add(m) or m
                for m in measures if m not in seen]

    # TODO: make sure resolve to valid measures
    not_found = set(measures) - keys(MEASURES)
    invalid = []
    for m in not_found:
        try:
            get_measure(m)
        except Exception:
            raise
            invalid.append(m)
    if invalid:
        raise ValueError('Could not resolve measures: '
                         '{}'.format(sorted(not_found)))

    if not incl_clustering:
        measures = [m for m in measures
                    if not get_measure(m).is_clustering]
    # TODO: remove clustering metrics given flag
    # raise error if empty
    if not measures:
        msg = 'Could not resolve {!r} to any measures.'.format(in_measures)
        if not incl_clustering:
            msg += ' Clustering measures have been excluded.'
        raise ValueError(msg)
    return measures


def get_measure(name):
    if isinstance(name, Measure):
        return name
    if name.count(':') == 2:
        return Measure.from_string(name)
    return MEASURES[name]


def get_measure_choices():
    return sorted(MEASURE_SETS.keys()) + sorted(MEASURES.keys())


MEASURE_HELP = ('Which measures to use: specify a name (or group name) from '
                'the list-measures command. This flag may be repeated.')


def _wrap(text):
    return '\n'.join(textwrap.wrap(text))


class ListMeasures(object):
    """List measures schemes available for evaluation"""

    def __init__(self, measures=None):
        self.measures = measures

    def __call__(self):
        measures = parse_measures(self.measures or get_measure_choices())
        header = ['Name', 'Aggregate', 'Filter', 'Key Fields', 'In groups']
        rows = [header]

        set_membership = defaultdict(list)
        for set_name, measure_set in sorted(MEASURE_SETS.items()):
            for name in parse_measures(measure_set):
                set_membership[name].append(set_name)

        for name in sorted(measures):
            measure = get_measure(name)
            rows.append((name, measure.agg, str(measure.filter),
                         '+'.join(measure.key),
                         ', '.join(set_membership[name])))

        col_widths = [max(len(row[i]) for row in rows)
                      for i in range(len(header))]
        rows.insert(1, ['=' * w for w in col_widths])
        fmt = '\t'.join('{:%ds}' % w for w in col_widths[:-1]) + '\t{}'
        ret = _wrap('The following lists possible values for --measure (-m) '
                    'in evaluate, confidence and significance. The name from '
                    'each row or the name of a group may be used. ') + '\n\n'
        ret = '\n'.join(textwrap.wrap(ret)) + '\n\n'
        ret += '\n'.join(fmt.format(*row) for row in rows)
        ret += '\n\nDefault evaluation group: {}'.format(DEFAULT_MEASURE_SET)
        ret += '\n\n'
        ret += _wrap('In all measures, a set of tuples corresponding to Key '
                     'Fields is produced from annotations matching Filter. '
                     'Aggregation with sets-micro compares gold and predicted '
                     'tuple sets directly; coreference aggregates compare '
                     'tuples clustered by their assigned entity ID.')
        ret += '\n\n'
        ret += ('A measure may be specified explicitly. Thus:\n'
                '  {}\nmay be entered as\n  {}'
                ''.format(DEFAULT_MEASURE, get_measure(DEFAULT_MEASURE)))
        return ret

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('-m', '--measure', dest='measures', action='append',
                       metavar='NAME', help=MEASURE_HELP)
        p.set_defaults(cls=cls)
        return p
