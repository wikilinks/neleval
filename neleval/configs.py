import textwrap
from collections import defaultdict
from .annotation import Measure, Annotation

try:
    keys = dict.viewkeys
except Exception:
    # Py3k
    keys = dict.keys


MEASURES = {
    # Mention evaluation measures
    'strong_mention_match':         Measure(['span']),
    'strong_typed_mention_match':   Measure(['span', 'type']),
    'strong_linked_mention_match':  Measure(['span'], 'is_linked'),
    # Linking evaluation measures
    'strong_link_match':            Measure(['span', 'kbid'], 'is_linked'),
    'strong_nil_match':             Measure(['span'], 'is_nil'),
    'strong_all_match':             Measure(['span', 'kbid']),
    'strong_typed_link_match':      Measure(['span', 'type', 'kbid'],
                                            'is_linked'),
    'strong_typed_nil_match':       Measure(['span', 'type'], 'is_nil'),
    'strong_typed_all_match':       Measure(['span', 'type', 'kbid']),
    # Document-level tagging evaluation measures
    'entity_match':                 Measure(['docid', 'kbid'], 'is_linked'),
    # Clustering evaluation measures
    'muc':                          Measure(['span'], agg='muc'),
    'b_cubed':                      Measure(['span'], agg='b_cubed'),
    'b_cubed_plus':                 Measure(['span', 'kbid'], agg='b_cubed'),
    'entity_ceaf':                  Measure(['span'], agg='entity_ceaf'),
    'mention_ceaf':                 Measure(['span'], agg='mention_ceaf'),
    'mention_ceaf_plus':            Measure(['span', 'kbid'], agg='mention_ceaf'),
    'typed_mention_ceaf':           Measure(['span', 'type'], agg='mention_ceaf'),
    'typed_mention_ceaf_plus':      Measure(['span', 'type', 'kbid',], agg='mention_ceaf'),
    'pairwise':                     Measure(['span'], agg='pairwise'),
    # Cai & Strube (2010) evaluation measures
    #'cs_b_cubed':                   Measure(['span'], agg='cs_b_cubed'),
    #'entity_cs_ceaf':               Measure(['span'], agg='entity_cs_ceaf'),
    #'mention_cs_ceaf':              Measure(['span'], agg='mention_cs_ceaf'),
}


# Configuration constants
ALL_MEASURES = 'all'
ALL_TAGGING = 'all-tagging'
ALL_COREF = 'all-coref'
TAC09_MEASURES = 'tac09'   # used 2009-2010
TAC11_MEASURES = 'tac11'   # used 2011-2013
TAC14_MEASURES = 'tac14'   # used 2014-
TMP_MEASURES = 'tmp'
CORNOLTI_WWW13_MEASURES = 'cornolti'
HACHEY_ACL14_MEASURES = 'hachey'
LUO_MEASURES = 'luo'
CAI_STRUBE_MEASURES = 'cai'

MEASURE_SETS = {
    ALL_MEASURES: [
        ALL_TAGGING,
        ALL_COREF,
        ],
    ALL_TAGGING: {
        'strong_mention_match',
        'strong_typed_mention_match',
        'strong_linked_mention_match',
        'strong_link_match',
        'strong_nil_match',
        'strong_all_match',
        'strong_typed_link_match',
        'strong_typed_nil_match',
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
        'typed_mention_ceaf',
        'mention_ceaf_plus',
        'typed_mention_ceaf_plus',
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
    TAC09_MEASURES: [
        'strong_link_match',  # recall equivalent to kb accuracy
        'strong_nil_match',  # recall equivalent to nil accuracy
        'strong_all_match',  # equivalent to overall accuracy
        ],
    TAC11_MEASURES: [
        TAC09_MEASURES,
        'b_cubed', # standard b-cubed
        'b_cubed_plus', # also requires correct resolution
        ],
    TAC14_MEASURES: [
        TAC11_MEASURES,
        # Assess mention recognition in TAC 2014 end-to-end task
        'strong_mention_match', # span must match
        'strong_typed_mention_match', # span and type must match
        # Assess recognition and disambiguation in TAC 2014 end-to-end task
        'strong_typed_all_match',  # span, type and resolution/nil must match
        # Assess recognition and clustering in TAC 2014 end-to-end task
        'mention_ceaf', # prf based on cluster alignment
        'typed_mention_ceaf',  # same requiring type match
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


def parse_measures(in_measures, incl_clustering=True, allow_unknown=False):
    """Expand measure lists and validate, but do not map strings to Measures

    """
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
            invalid.append(m)
    if invalid and not allow_unknown:
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


def get_measure(name, weighting=None):
    if isinstance(name, Measure):
        measure = name
    elif name.count(':') == 2:
        measure = Measure.from_string(name)
    else:
        measure = MEASURES[name]
    return measure.with_weighting(weighting)


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
        from .coref_metrics import COREF_METRICS
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
                     'Aggregation with `sets\' compares gold and predicted '
                     'tuple sets directly; coreference aggregates compare '
                     'tuples clustered by their assigned entity ID.')
        ret += '\n\n'
        ret += ('A measure may be specified explicitly. Thus:\n'
                '  {}\nmay be entered as\n  {}'
                ''.format(DEFAULT_MEASURE, get_measure(DEFAULT_MEASURE)))
        ret += '\n\n'
        ret += ('Available aggregates are:\n'
                '- non-clustering: %s\n'
                '- clustering: %s' % (', '.join(sorted(Measure.NON_CLUSTERING_AGG)),
                                      ', '.join(sorted(COREF_METRICS))))
        ret += '\n\n'
        ret += _wrap('Available filter and key fields: %s.' %
                     ', '.join(sorted(Annotation.list_fields())))
        ret += '\n\n'
        ret += _wrap('More fields can be stored dynamically by entering a candidate\'s type as a JSON key-value mapping.')
        return ret

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('-m', '--measure', dest='measures', action='append',
                       metavar='NAME', help=MEASURE_HELP)
        p.set_defaults(cls=cls)
        return p


class TypeWeighting:
    def __init__(self, path):
        self.values = {}
        with open(path) as f:
            for l in f:
                gold, sys, weight = l.split('\t')
                weight = float(weight)
                self.values[gold, sys] = max(weight,
                                             self.values.get((gold, sys),
                                                             weight))

    def __call__(self, gold_type, sys_type):
        return self.values.get((gold_type, sys_type),
                               int(gold_type == sys_type))


def load_weighting(type_weights=None):
    """type_weights is a path
    """
    weighting = {}
    if type_weights is not None:
        weighting['type'] = TypeWeighting(type_weights)
    return weighting or None
