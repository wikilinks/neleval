from neleval.configs import LUO_MEASURES, parse_measures
from neleval.coref_metrics import mapping_to_sets, sets_to_mapping
from neleval.coref_metrics import _prf, muc
from neleval import coref_metrics

from neleval.tests.util import check_correct

MAPPING = {'a': 1, 'b': 2, 'c': 1}
SETS = {1: {'a', 'c'}, 2: {'b'}}


def test_conversions():
    assert mapping_to_sets(MAPPING) == SETS
    assert sets_to_mapping(SETS) == MAPPING
    assert sets_to_mapping(mapping_to_sets(MAPPING)) == MAPPING
    assert mapping_to_sets(sets_to_mapping(SETS)) == SETS


def _get_coref_fscore(gold, resp, measures):
    for name in parse_measures(measures):
        f = getattr(coref_metrics, name)
        yield f.__name__, round(_prf(*f(gold, resp))[2], 3)

# TC-A-* tests from https://code.google.com/p/reference-coreference-scorers
RCS14_TCA_GOLD = {'0': {1}, '1': {2,3}, '2': {4,5,6}}
RCS14_TCA_RESPS = [
    ('TC-A-1', # perfect
     {'0': {1}, '1': {2,3}, '2': {4,5,6}},
     {'muc': 1.0, 'b_cubed': 1.0,
      'mention_ceaf': 1.0, 'entity_ceaf': 1.0}),
    ('TC-A-2', # response with missing mentions/entities
     {'0': {1}, '2': {4,5}},
     {'muc': 0.5, 'b_cubed': 0.56,
      'mention_ceaf': 0.667, 'entity_ceaf': 0.72}),
    ('TC-A-3', # response with false-alarm mentions/entities
     {'0': {1}, '1': {2,3,7}, '2': {4,5,6,8}, '3': {9}},
     {'muc': 0.75, 'b_cubed': 0.675,
      'mention_ceaf': 0.8, 'entity_ceaf': 0.759}),
    ('TC-A-4', # response with both missing and false-alarm mentions/entities
     {'0': {1}, '1': {2,3,7}, '2': {4,8}, '3': {9}},
     {'muc': 0.333, 'b_cubed': 0.468,
      'mention_ceaf': 0.615, 'entity_ceaf': 0.629}),
    # NOTE TC-A-5 through TC-A-9 test IO, not metrics
#    # TODO Verify TC-A-10 ceafm and ceafe values
#    ('TC-A-10', # Gold mentions. Only singletons in the response.
#     {'0': {1}, '1': {2}, '2': {3}, '3': {4}, '4': {5}, '5': {6}},
#     {'muc': 0.0, 'b_cubed': 0.667,
#      'mention_ceaf': 0.5, 'entity_ceaf': 0.481}),
#    # TODO Verify TC-A-11 ceafm and ceafe values
#    ('TC-A-11', # Gold mentions. All mentions are coreferent in the response.
#     {'0': {1,2,3,4,5,6}},
#     {'muc': 0.75, 'b_cubed': 0.56,
#      'mention_ceaf': 0.5, 'entity_ceaf': 0.333}),
#    # TODO Verify TC-A-12 ceafm and ceafe values
#    ('TC-A-12', # System mentions. Only singletons in the response.
#     {'0': {1}, '1': {7}, '2': {8}, '3': {3}, '4': {4}, '5': {5}, '6': {9}},
#     {'muc': 0.0, 'b_cubed': 0.443,
#      'mention_ceaf': 0.462, 'entity_ceaf': 0.433}),
#    # TODO Verify TC-A-13 ceafm and ceafe values
#    ('TC-A-13', # System mentions. All mentions are coreferent in the response.
#     {'0': {1,7,8,3,4,5,9}},
#     {'muc': 0.222, 'b_cubed': 0.194,
#      'mention_ceaf': 0.308, 'entity_ceaf': 0.2}),
    ]

def test_rcs_tca_ceaf():
    "Examples from Luo (2005)"
    for system, response, expected in RCS14_TCA_RESPS:
        actual = dict(_get_coref_fscore(RCS14_TCA_GOLD, response, LUO_MEASURES))
        check_correct(expected, actual)

## TC-B test from https://code.google.com/p/reference-coreference-scorers
#RCS14_TCB_GOLD = {'10043': {1,2}, '10054': {3,4,5}}
#RCS14_TCB_RESPS = [
#    # TODO Verify TC-B-1 muc, b_cubed, ceafm and ceafe values
#    ('TC-B-1', # spurious mention (x) and missing mention (a) in response; link (bc) is a key non-coref link and is an incorrect response coref link.
#     {'10043': {2,3,6}, '10054': {4,5}},
#     {'muc': 0.333, 'b_cubed': 0.478,
#      'mention_ceaf': 0.6, 'entity_ceaf': 0.6}),
#    ]
#
#def test_rcs_tcb_ceaf():
#    "Examples from Luo (2005)"
#    for system, response, expected in RCS14_TCB_RESPS:
#        actual = dict(_get_coref_fscore(RCS14_TCB_GOLD, response, LUO_MEASURES))
#        check_correct(expected, actual)

## TC-C test from https://code.google.com/p/reference-coreference-scorers
#RCS14_TCC_GOLD = {'10043': {1,2}, '10054': {3,4,5}, '10060': {6,7}}
#RCS14_TCC_RESPS = [
#    # TODO Verify TC-C-1 muc, b_cubed, ceafm and ceafe values
#    ('TC-C-1', # plus a new entity and its correct prediction shown. this was for testing the more than two entity case
#     {'10043': {2,3,6}, '10054': {4,5}, '10060': {6,7}},
#     {'muc': 0.5, 'b_cubed': 0.674,
#      'mention_ceaf': 0.714, 'entity_ceaf': 0.733}),
#    ]
#
#def test_rcs_tcc_ceaf():
#    "Examples from Luo (2005)"
#    for system, response, expected in RCS14_TCC_RESPS:
#        actual = dict(_get_coref_fscore(RCS14_TCC_GOLD, response, LUO_MEASURES))
#        check_correct(expected, actual)

# TC-M test from https://code.google.com/p/reference-coreference-scorers
RCS14_TCM_GOLD = {'0': {1,2,3,4,5,6}}
RCS14_TCM_RESPS = [
    ('TC-M-1',
     {'0': {1,2,3,4,5,6}},
     {'muc': 1.0, 'b_cubed': 1.0,
      'mention_ceaf': 1.0, 'entity_ceaf': 1.0}),
#    # TODO Verify TC-M-2 muc, b_cubed, ceafm and ceafe values
#    ('TC-M-2',
#     {'0': {1}, '1': {2}, '2': {3}, '3': {4}, '4': {5}, '5': {6}},
#     {'muc': 0.0, 'b_cubed': 0.286,
#      'mention_ceaf': 0.167, 'entity_ceaf': 0.082}),
#    # TODO Verify TC-M-3 muc, b_cubed, ceafm and ceafe values
#    ('TC-M-3',
#     {'0': {1,2}, '1': {3,4,5}, '2': {6}},
#     {'muc': 0.75, 'b_cubed': 0.56,
#      'mention_ceaf': 0.5, 'entity_ceaf': 0.333}),
#    # TODO Verify TC-M-4 muc, b_cubed, ceafm and ceafe valuesw
#    ('TC-M-4',
#     {'0': {1,2,3,7,8,9}},
#     {'muc': 0.4, 'b_cubed': 0.25,
#      'mention_ceaf': 0.5, 'entity_ceaf': 0.5}),
#    # TODO Verify TC-M-5 b_cubed, ceafm and ceafe valuesw
#    ('TC-M-5',
#     {'0': {1}, '1': {2}, '2': {3}, '3': {7}, '4': {8}, '5': {9}},
#     {'muc': 0.0, 'b_cubed': 0.143,
#      'mention_ceaf': 0.167, 'entity_ceaf': 0.082}),
#    # TODO Verify TC-M-6 muc, b_cubed, ceafm and ceafe valuesw
#    ('TC-M-6',
#     {'0': {1,2}, '1': {3,7,8}, '2': {9}},
#     {'muc': 0.25, 'b_cubed': 0.205,
#      'mention_ceaf': 0.333, 'entity_ceaf': 0.25}),
    ]

def test_rcs_tcc_ceaf():
    "Examples from Luo (2005)"
    for system, response, expected in RCS14_TCM_RESPS:
        actual = dict(_get_coref_fscore(RCS14_TCM_GOLD, response, LUO_MEASURES))
        check_correct(expected, actual)

## TC-N test from https://code.google.com/p/reference-coreference-scorers
#RCS14_TCN_GOLD = {'0': {1}, '1': {2}, '2': {3}, '3': {4}, '4': {5}, '5': {6}}
#RCS14_TCN_RESPS = [
#    ('TC-N-1',
#     {'0': {1}, '1': {2}, '2': {3}, '3': {4}, '4': {5}, '5': {6}},
#     {'muc': 0.0, 'b_cubed': 1.0,
#      'mention_ceaf': 1.0, 'entity_ceaf': 1.0}),
#    # TODO Verify TC-N-2 muc, b_cubed, ceafm and ceafe values
#    ('TC-N-2',
#     {'0': {1,2,3,4,5,6}},
#     {'muc': 0.0, 'b_cubed': 0.286,
#      'mention_ceaf': 0.167, 'entity_ceaf': 0.082}),
#    # TODO Verify TC-N-3 muc, b_cubed, ceafm and ceafe values
#    ('TC-N-3',
#     {'0': {1,2}, '1': {3,4,5}, '2': {6}},
#     {'muc': 0.0, 'b_cubed': 0.667,
#      'mention_ceaf': 0.5, 'entity_ceaf': 0.481}),
#    # TODO Verify TC-N-4 b_cubed, ceafm and ceafe valuesw
#    ('TC-N-4',
#     {'0': {1}, '1': {2}, '2': {3}, '3': {7}, '4': {8}, '5': {9}},
#     {'muc': 0.0, 'b_cubed': 0.5,
#      'mention_ceaf': 0.5, 'entity_ceaf': 0.5}),
#    # TODO Verify TC-N-5 b_cubed, ceafm and ceafe valuesw
#    ('TC-N-5',
#     {'0': {1,2,3,7,8,9}},
#     {'muc': 0.0, 'b_cubed': 0.143,
#      'mention_ceaf': 0.167, 'entity_ceaf': 0.082}),
#    # TODO Verify TC-N-6 muc, b_cubed, ceafm and ceafe valuesw
#    ('TC-N-6',
#     {'0': {1,2}, '1': {3,7,8}, '2': {9}},
#     {'muc': 0.0, 'b_cubed': 0.308,
#      'mention_ceaf': 0.333, 'entity_ceaf': 0.259}),
#    ]
#
#def test_rcs_tcc_ceaf():
#    "Examples from Luo (2005)"
#    for system, response, expected in RCS14_TCN_RESPS:
#        actual = dict(_get_coref_fscore(RCS14_TCN_GOLD, response, LUO_MEASURES))
#        check_correct(expected, actual)

LUO05_GOLD = {'A': {1,2,3,4,5}, 'B': {6,7}, 'C': {8,9,10,11,12}}
LUO05_RESPS = [
    ('sysa',
     {'A': {1,2,3,4,5}, 'B': {6,7,8,9,10,11,12}},
     {'muc': 0.947, 'b_cubed': 0.865,
      'mention_ceaf': 0.833, 'entity_ceaf': 0.733}),
    ('sysb',
     {'A': {1,2,3,4,5,8,9,10,11,12}, 'B': {6,7}},
     {'muc': 0.947, 'b_cubed': 0.737,
      'mention_ceaf': 0.583, 'entity_ceaf': 0.667}),
    ('sysc',
     {'A': {1,2,3,4,5,6,7,8,9,10,11,12}},
     {'muc': 0.900, 'b_cubed': 0.545,
      'mention_ceaf': 0.417, 'entity_ceaf': 0.294}),
    ('sysd',
     {i: {i,} for i in range(1,13)},
     {'muc': 0.0, 'b_cubed': 0.400, 
      'mention_ceaf': 0.250, 'entity_ceaf': 0.178})
    ]
def test_luo_ceaf():
    "Examples from Luo (2005)"
    for system, response, expected in LUO05_RESPS:
        actual = dict(_get_coref_fscore(LUO05_GOLD, response, LUO_MEASURES))
        check_correct(expected, actual)

def _get_muc_prf(gold, resp):
    return tuple(round(v, 3) for v in _prf(*muc(gold, resp)))

VILAIN95 = [
    # Table 1, Row 1
    ({1: {'A', 'B', 'C', 'D'}},
     {1: {'A', 'B'}, 2: {'C', 'D'}},
     (1.0, 0.667, 0.8)),
    # Table 1, Row 2
    ({1: {'A', 'B'}, 2: {'C', 'D'}},
     {1: {'A', 'B', 'C', 'D'}},
     (0.667, 1.0, 0.8)),
    # Table 1, Row 3
    ({1: {'A', 'B', 'C', 'D'}},
     {1: {'A', 'B', 'C', 'D'}},
     (1.0, 1.0, 1.0)),
    # Table 1, Row 4
    ({1: {'A', 'B', 'C', 'D'}},
     {1: {'A', 'B'}, 2: {'C', 'D'}},
     (1.0, 0.667, 0.8)),
    # Table 1, Row 5
    ({1: {'A', 'B', 'C'}},
     {1: {'A', 'C'}},
     (1.0, 0.5, 0.667)),
    # More complex 1
    ({1: {'B', 'C', 'D', 'E', 'G', 'H', 'J'}},
     {1: {'A', 'B', 'C'}, 2: {'D', 'E', 'F'}, 3: {'G', 'H', 'I'}},
     (0.5, 0.5, 0.5)),
    # More complex 2
    ({1: {'A', 'B', 'C'}, 2: {'D', 'E', 'F', 'G'}},
     {1: {'A', 'B'}, 2: {'C', 'D'}, 3: {'F', 'G', 'H'}},
     (0.5, 0.4, 0.444)),
    ]
def test_vilain_muc():
    "Examples from Vilain et al. (1995)"
    for key, response, expected in VILAIN95:
        assert _get_muc_prf(key, response) == expected


CAI10_TABLES_4_5 = [
    ({1: {'a', 'b', 'c'}},    # true
     {2: {'a', 'b'}, 3: {'c'}, 4: {'i'}, 5: {'j'}},   # pred
     {'cs_b_cubed': (1.0, 0.556, 0.714),  # Note paper says 0.715, but seems incorrect
      'mention_cs_ceaf': (0.667, 0.667, 0.667)}),
    ({1: {'a', 'b', 'c'}},
     {2: {'a', 'b'}, 3: {'i', 'j'}, 4: {'c'}},
     {'cs_b_cubed': (.8, .556, .656),
      'mention_cs_ceaf': (.6, .667, .632)}),
    ({1: {'a', 'b', 'c'}},
     {2: {'a', 'b'}, 3: {'i', 'j'}, 4: {'k', 'l'}, 5: {'c'}},
     {'cs_b_cubed': (.714, .556, .625),
      'mention_cs_ceaf': (.571, .667, .615)}),
    ({1: {'a', 'b', 'c'}},
     {2: {'a', 'b'}, 3: {'i', 'j', 'k', 'l'}},
     {'cs_b_cubed': (.571, .556, .563),
      'mention_cs_ceaf': (.429, .667, .522)}),
]


###def test_cai_strube_twinless_adjustment():
###    "Examples from Cai & Strube (SIGDIAL'10)"
###    for true, pred, expected in CAI10_TABLES_4_5:
###        actual = {f: tuple(round(x, 3) for x in _prf(*getattr(coref_metrics, f)(true, pred)))
###                  for f in parse_measures(CAI_STRUBE_MEASURES)}
###        check_correct(expected, actual)

