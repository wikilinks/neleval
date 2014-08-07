import textwrap
from collections import defaultdict
from .annotation import Matcher

try:
    keys = dict.viewkeys
except Exception:
    # Py3k
    keys = dict.keys


MATCHERS = {
    'strong_mention_match':         Matcher(['span']),
    'strong_linked_mention_match':  Matcher(['span'], 'is_linked'),
    'strong_link_match':            Matcher(['span', 'kbid'], 'is_linked'),
    'strong_nil_match':             Matcher(['span'], 'is_nil'),
    'strong_all_match':             Matcher(['span', 'kbid']),
    'strong_typed_all_match':       Matcher(['span', 'type', 'kbid']),
    'entity_match':                 Matcher(['docid', 'kbid'], 'is_linked'),

    'b_cubed_plus':                 Matcher(['span', 'kbid'], agg='b_cubed'),
}

for name in ['muc', 'b_cubed', 'entity_ceaf', 'mention_ceaf', 'pairwise',
             #'cs_b_cubed', 'entity_cs_ceaf', 'mention_cs_ceaf']:
             ]:
    MATCHERS[name] = Matcher(['span'], agg=name)


# Configuration constants
ALL_MATCHES = 'all'
ALL_TAGGING = 'all-tagging'
ALL_COREF = 'all-coref'
TAC_MATCHES = 'tac'
TAC14_MATCHES = 'tac14'
TMP_MATCHES = 'tmp'
CORNOLTI_WWW13_MATCHES = 'cornolti'
HACHEY_ACL14_MATCHES = 'hachey'
LUO_MATCHES = 'luo'
CAI_STRUBE_MATCHES = 'cai'

MATCH_SETS = {
    ALL_MATCHES: [
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
        'entity_match',
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
    CORNOLTI_WWW13_MATCHES: [
        'strong_linked_mention_match',
        'strong_link_match',
        'entity_match',
        ],
    HACHEY_ACL14_MATCHES: [
        'strong_mention_match',  # full ner
        'strong_linked_mention_match',
        'strong_link_match',
        'entity_match',
        ],
    LUO_MATCHES: [
        'muc',
        'b_cubed',
        'mention_ceaf',
        'entity_ceaf',
        ],
    #CAI_STRUBE_MATCHES: [
    #    'cs_b_cubed',
    #    'entity_cs_ceaf',
    #    'mention_cs_ceaf',
    #],
    TAC_MATCHES: [
        'strong_link_match',  # recall equivalent to kb accuracy before 2014
        'strong_nil_match',  # recall equivalent to nil accuracy before 2014
        'strong_all_match',  # equivalent to overall accuracy before 2014
        'strong_typed_all_match',  # wikification f-score for TAC 2014

        'mention_ceaf',
        'b_cubed',
        'b_cubed_plus',
        ],
    TAC14_MATCHES: [
        'strong_typed_all_match',  # wikification f-score for TAC 2014
    ],
    TMP_MATCHES: [
        'mention_ceaf',
        'entity_ceaf',
        'pairwise',
        ],
}

DEFAULT_MATCH_SET = ALL_MATCHES
DEFAULT_MATCH = 'strong_all_match'


def _expand(matches):
    if isinstance(matches, str):
        if matches in MATCH_SETS:
            matches = MATCH_SETS[matches]
        else:
            return [matches]
    if isinstance(matches, Matcher):
        return [Matcher]
    if len(matches) == 1:
        return _expand(matches[0])
    return [m for group in matches for m in _expand(group)]


def parse_matches(in_matches, incl_clustering=True):
    # flatten nested sequences and expand group names
    matches = _expand(in_matches)
    # remove duplicates while maintaining order
    seen = set()
    matches = [seen.add(m) or m
               for m in matches if m not in seen]

    # TODO: make sure resolve to valid matchers
    not_found = set(matches) - keys(MATCHERS)
    invalid = []
    for m in not_found:
        try:
            get_matcher(m)
        except Exception:
            raise
            invalid.append(m)
    if invalid:
        raise ValueError('Could not resolve matchers: {}'.format(sorted(not_found)))

    if not incl_clustering:
        matches = [m for m in matches
                   if not get_matcher(m).is_clustering_match]
    # TODO: remove clustering metrics given flag
    # raise error if empty
    if not matches:
        msg = 'Could not resolve {!r} to any matches.'.format(in_matches)
        if not incl_clustering:
            msg += ' Clustering metrics have been excluded.'
        raise ValueError(msg)
    return matches


def get_matcher(name):
    if isinstance(name, Matcher):
        return name
    if name.count(':') == 2:
        return Matcher.from_string(name)
    return MATCHERS[name]


def get_match_choices():
    return sorted(MATCH_SETS.keys()) + sorted(MATCHERS.keys())


MATCH_HELP = ('Which metrics to use: specify a name (or group name) from the '
              'list-metrics command. This flag may be repeated.')


def _wrap(text):
    return '\n'.join(textwrap.wrap(text))


class ListMetrics(object):
    """List matching schemes available for evaluation"""

    def __init__(self, matches=None):
        self.matches = matches

    def __call__(self):
        matches = parse_matches(self.matches or get_match_choices())
        header = ['Name', 'Aggregate', 'Filter', 'Key Fields', 'In groups']
        rows = [header]

        set_membership = defaultdict(list)
        for set_name, match_set in sorted(MATCH_SETS.items()):
            for name in parse_matches(match_set):
                set_membership[name].append(set_name)

        for name in sorted(matches):
            matcher = get_matcher(name)
            rows.append((name, matcher.agg, str(matcher.filter),
                         '+'.join(matcher.key),
                         ', '.join(set_membership[name])))

        col_widths = [max(len(row[i]) for row in rows)
                      for i in range(len(header))]
        rows.insert(1, ['=' * w for w in col_widths])
        fmt = '\t'.join('{:%ds}' % w for w in col_widths[:-1]) + '\t{}'
        ret = _wrap('The following lists possible values for --match (-m) in '
                    'evaluate, confidence and significance. The name from '
                    'each row or the name of a group may be used. ') + '\n\n'
        ret = '\n'.join(textwrap.wrap(ret)) + '\n\n'
        ret += '\n'.join(fmt.format(*row) for row in rows)
        ret += '\n\nDefault evaluation group: {}'.format(DEFAULT_MATCH_SET)
        ret += '\n\n'
        ret += _wrap('In all metrics, a set of tuples corresponding to Key '
                     'Fields is produced from annotations matching Filter. '
                     'Aggregation with sets-micro compares gold and predicted '
                     'tuple sets directly; coreference aggregates compare '
                     'tuples clustered by their assigned entity ID.')
        ret += '\n\n'
        ret += ('A metric may be specified explicitly. Thus:\n'
                '  {}\nmay be entered as\n  {}'
                ''.format(DEFAULT_MATCH, get_matcher(DEFAULT_MATCH)))
        return ret

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('-m', '--match', dest='matches', action='append',
                       metavar='NAME', help=MATCH_HELP)
        p.set_defaults(cls=cls)
        return p
