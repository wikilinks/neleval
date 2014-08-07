#!/usr/bin/env python
from .document import Reader as AnnotationReader
from .data import Reader, Mention, Writer
from .configs import MATCH_SETS, TMP_MATCHES, LUO_MATCHES, CAI_STRUBE_MATCHES, ALL_MATCHES
from .coref_metrics import mapping_to_sets, sets_to_mapping
from .coref_metrics import _prf, muc
from . import coref_metrics
from .evaluate import Evaluate
from .formats import Unstitch, Stitch
from .tac import PrepareTac
from .utils import normalise_link
from io import BytesIO
from pprint import pprint
import os

from nose.tools import assert_sequence_equal

DIR = os.path.join(os.path.dirname(__file__))
EXAMPLES = os.path.join(DIR, 'examples')
CONLL_GOLD = os.path.join(EXAMPLES, 'conll_gold.txt')
CONLL_GOLD_UNSTITCHED = os.path.join(EXAMPLES, 'conll_gold.unstitched.tsv')
CONLL_SYSA = os.path.join(EXAMPLES, 'conll_sysa.txt')
CONLL_SYSA_UNSTITCHED = os.path.join(EXAMPLES, 'conll_sysa.unstitched.tsv')
CONLL_MULTISENT = os.path.join(EXAMPLES, 'conll_multisent.txt')
CONLL_MULTI_GOLD = os.path.join(EXAMPLES, 'conll_multi_gold.txt')
CONLL_MULTI_GOLD_UNSTITCHED = os.path.join(EXAMPLES,
                                           'conll_multi_gold.unstitched.tsv')
CONLL_MULTI_SYSA = os.path.join(EXAMPLES, 'conll_multi_sysa.txt')
CONLL_MULTI_SYSA_UNSTITCHED = os.path.join(EXAMPLES,
                                           'conll_multi_sysa.unstitched.tsv')
TAC_GOLD_QUERIES = os.path.join(EXAMPLES, 'tac_gold.xml')
TAC_GOLD_LINKS = os.path.join(EXAMPLES, 'tac_gold.tab')
TAC_GOLD_COMB = os.path.join(EXAMPLES, 'tac_gold.combined.tsv')
TAC_SYS_QUERIES = os.path.join(EXAMPLES, 'tac_system.xml')
TAC_SYS_LINKS = os.path.join(EXAMPLES, 'tac_system.tab')
TAC_SYS_COMB = os.path.join(EXAMPLES, 'tac_system.combined.tsv')

# PREPARE TESTS

def test_tac_prepare():
    for linkf, queryf, preparedf in (
        (TAC_GOLD_LINKS, TAC_GOLD_QUERIES, TAC_GOLD_COMB),
        (TAC_SYS_LINKS, TAC_SYS_QUERIES, TAC_SYS_COMB),
        ):
        prepared = PrepareTac(linkf, queryf)()
        assert prepared == open(preparedf).read().rstrip('\n')

def conll_unstitch(f):
    u_fname = '{}.unstitched.tmp'.format(f)
    with open(u_fname, 'w') as u_fh:
        print >>u_fh, Unstitch(f)()
    return u_fname

def test_conll_unstitch():
    unstitched = Unstitch(CONLL_GOLD)()
    assert unstitched == open(CONLL_GOLD_UNSTITCHED).read().rstrip('\n')

def test_conll_unstitch_stitch():
    for f in (CONLL_GOLD, CONLL_MULTISENT):
        u_fname = conll_unstitch(f)
        s_str = Stitch(u_fname, f)()
        s_str = '\n'.join(l.rstrip('\t') for l in s_str.split('\n'))
        assert s_str == open(f).read()
        os.remove(u_fname)

# READ/WRITE TESTS

def test_annotation_read_write():
    docs = list(AnnotationReader(open(TAC_GOLD_COMB)))
    d_str = '\n'.join([str(d) for d in docs])
    assert d_str == open(TAC_GOLD_COMB).read().rstrip('\n')

def test_conll_data():
    d = list(Reader(open(CONLL_GOLD)))
    assert len(d) == 1
    doc = list(d)[0] 
    assert len(list(doc.iter_mentions())) == 2
    assert len(list(doc.iter_links())) == 1
    assert len(list(doc.iter_nils())) == 1

def test_conll_read_write():
    for f in (CONLL_GOLD, CONLL_MULTISENT):
        out = BytesIO()
        w = Writer(out)
        for doc in list(Reader(open(f))):
            w.write(doc)
        w_str = '\n'.join(l.rstrip('\t') for l in out.getvalue().split('\n'))
        assert w_str == open(f).read()

def test_sentences():
    """ Checks that we can read contiguous sentences with the indices making sense. """
    docs = list(Reader(open(CONLL_MULTISENT)))
    for d in docs:
        last = None
        for s in d.sentences:
            for span in s:
                if last:
                    assert span.start == last
                else:
                    assert span.start == 0
                last = span.end
                if isinstance(span, Mention) and span.link is not None:
                    assert isinstance(span.score, float)

# WIKIPEDIA TITLE NORM TESTS

def test_normalisation():
    assert normalise_link('Some title') == 'Some_title'
    assert normalise_link('http://en.wikipedia.org/wiki/Some title') == 'Some_title'
    assert normalise_link('http://fr.wikipedia.org/wiki/Some') == 'Some'
    assert normalise_link('Some') == 'Some'

# EVAL TEST UTILITIES

def check_correct(expected, actual):
    assert expected.viewkeys() == actual.viewkeys(), 'Different keys\nexpected\t{}\nactual\t{}'.format(sorted(expected.keys()), sorted(actual.keys()))
    for k in expected:
        exp = expected[k]
        act = actual[k]
        if hasattr(exp, '__iter__') and not hasattr(exp, 'items'):
            assert_sequence_equal(exp, act, 'Different on key "{}".\nexpected\t{}\nactual\t{}'.format(k, exp, act))
        else:
            assert exp == act, 'Different on key "{}".\nexpected\t{}\nactual\t{}'.format(k, exp, act)
    return True

# CLUSTER EVAL TESTS

MAPPING = {'a': 1, 'b': 2, 'c': 1}
SETS = {1: {'a', 'c'}, 2: {'b'}}
def test_conversions():
    assert mapping_to_sets(MAPPING) == SETS
    assert sets_to_mapping(SETS) == MAPPING
    assert sets_to_mapping(mapping_to_sets(MAPPING)) == MAPPING
    assert mapping_to_sets(sets_to_mapping(SETS)) == SETS
    

def _get_coref_fscore(gold, resp, matches):
    for name in parse_matches(matches):
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
        actual = dict(_get_coref_fscore(RCS14_TCA_GOLD, response, LUO_MATCHES))
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
#        actual = dict(_get_coref_fscore(RCS14_TCB_GOLD, response, LUO_MATCHES))
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
#        actual = dict(_get_coref_fscore(RCS14_TCC_GOLD, response, LUO_MATCHES))
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
        actual = dict(_get_coref_fscore(RCS14_TCM_GOLD, response, LUO_MATCHES))
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
#        actual = dict(_get_coref_fscore(RCS14_TCN_GOLD, response, LUO_MATCHES))
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
        actual = dict(_get_coref_fscore(LUO05_GOLD, response, LUO_MATCHES))
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


def test_cai_strube_twinless_adjustment():
    "Examples from Cai & Strube (SIGDIAL'10)"
    for true, pred, expected in CAI10_TABLES_4_5:
        actual = {f: tuple(round(x, 3) for x in _prf(*getattr(coref_metrics, f)(true, pred)))
                  for f in parse_matches(CAI_STRUBE_MATCHES)]}
        check_correct(expected, actual)


# EVALUATE TESTS

def _get_stats(gold_path, sys_path):
    stats = Evaluate(sys_path, gold=gold_path,
                     matches=ALL_MATCHES,  # TODO add test output for all
                     fmt='none')()
    pprint(stats)
    return stats

EXPECTED_TAC_SYS = {
 'entity_match': {'fn': 0,
                  'fp': 0,
                  'fscore': 1.0,
                  'precision': 1.0,
                  'recall': 1.0,
                  'ptp': 6,
                  'rtp': 6},
 'strong_link_match': {'fn': 0,
                       'fp': 0,
                       'fscore': 1.0,
                       'precision': 1.0,
                       'recall': 1.0,
                       'ptp': 6,
                       'rtp': 6},
 'strong_nil_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 4,
                      'rtp': 4},
 'strong_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 10,
                      'rtp': 10},
 'strong_typed_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 10,
                      'rtp': 10},
 'strong_mention_match': {'fn': 0,
                          'fp': 0,
                          'fscore': 1.0,
                          'precision': 1.0,
                          'recall': 1.0,
                          'ptp': 10,
                          'rtp': 10},
 'strong_linked_mention_match': {'fn': 0,
                                 'fp': 0,
                                 'fscore': 1.0,
                                 'precision': 1.0,
                                 'recall': 1.0,
                                 'ptp': 6,
                                 'rtp': 6},
 'entity_ceaf': {'fn': 0.0,
                 'fp': 0.0,
                 'fscore': 1.0,
                 'precision': 1.0,
                 'ptp': 2.0,
                 'recall': 1.0,
                 'rtp': 2.0},
 'mention_ceaf': {'fn': 0.0,
                  'fp': 0.0,
                  'fscore': 1.0,
                  'precision': 1.0,
                  'ptp': 10.0,
                  'recall': 1.0,
                  'rtp': 10.0},
 'pairwise': {'fn': 0,
              'fp': 0,
              'fscore': 1.0,
              'precision': 1.0,
              'ptp': 19,
              'recall': 1.0,
              'rtp': 19},
}

def test_tac_eval():
    check_correct(EXPECTED_TAC_SYS, _get_stats(TAC_GOLD_COMB, TAC_SYS_COMB))

EXPECTED_CONLL_SELFEVAL = {
 'entity_match': {'fn': 0,
                       'fp': 0,
                       'fscore': 1.0,
                       'precision': 1.0,
                       'recall': 1.0,
                       'ptp': 1,
                       'rtp': 1},
 'strong_link_match': {'fn': 0,
                       'fp': 0,
                       'fscore': 1.0,
                       'precision': 1.0,
                       'recall': 1.0,
                       'ptp': 1,
                       'rtp': 1},
 'strong_nil_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 1,
                      'rtp': 1},
 'strong_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 2,
                      'rtp': 2},
 'strong_typed_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 2,
                      'rtp': 2},
 'strong_mention_match': {'fn': 0,
                          'fp': 0,
                          'fscore': 1.0,
                          'precision': 1.0,
                          'recall': 1.0,
                          'ptp': 2,
                          'rtp': 2},
 'strong_linked_mention_match': {'fn': 0,
                                 'fp': 0,
                                 'fscore': 1.0,
                                 'precision': 1.0,
                                 'recall': 1.0,
                                 'ptp': 1,
                                 'rtp': 1},
 'entity_ceaf': {'fn': 0.0,
                 'fp': 0.0,
                 'fscore': 1.0,
                 'precision': 1.0,
                 'ptp': 1.0,
                 'recall': 1.0,
                 'rtp': 1.0},
 'mention_ceaf': {'fn': 0.0,
                  'fp': 0.0,
                  'fscore': 1.0,
                  'precision': 1.0,
                  'ptp': 2.0,
                  'recall': 1.0,
                  'rtp': 2.0},
 'pairwise': {'fn': 0,
              'fp': 0,
              'fscore': 1.0,
              'precision': 1.0,
              'ptp': 2,
              'recall': 1.0,
              'rtp': 2},
}

def test_conll_selfeval():
    assert check_correct(EXPECTED_CONLL_SELFEVAL,
                         _get_stats(CONLL_GOLD_UNSTITCHED,
                                    CONLL_GOLD_UNSTITCHED))

EXPECTED_CONLL_SYSA = {
 'entity_match': {'fn': 0,
                       'fp': 1,
                       'fscore': 0.6666666666666666,
                       'precision': 0.5,
                       'recall': 1.0,
                       'ptp': 1,
                       'rtp': 1},
 'strong_link_match': {'fn': 0,
                       'fp': 1,
                       'fscore': 0.6666666666666666,
                       'precision': 0.5,
                       'recall': 1.0,
                       'ptp': 1,
                       'rtp': 1},
 'strong_nil_match': {'fn': 1,
                      'fp': 0,
                      'fscore': 0.0,
                      'precision': 0.0,
                      'recall': 0.0,
                      'ptp': 0,
                      'rtp': 0},
 'strong_all_match': {'fn': 1,
                      'fp': 1,
                      'fscore': 0.5,
                      'precision': 0.5,
                      'recall': 0.5,
                      'ptp': 1,
                      'rtp': 1},
 'strong_typed_all_match': {'fn': 1,
                      'fp': 1,
                      'fscore': 0.5,
                      'precision': 0.5,
                      'recall': 0.5,
                      'ptp': 1,
                      'rtp': 1},
 'strong_mention_match': {'fn': 0,
                          'fp': 0,
                          'fscore': 1.0,
                          'precision': 1.0,
                          'recall': 1.0,
                          'ptp': 2,
                          'rtp': 2},
 'strong_linked_mention_match': {'fn': 0,
                                 'fp': 1,
                                 'fscore': 0.6666666666666666,
                                 'precision': 0.5,
                                 'recall': 1.0,
                                 'ptp': 1,
                                 'rtp': 1},
 'entity_ceaf': {'fn': 0.0,
                 'fp': 0.0,
                 'fscore': 1.0,
                 'precision': 1.0,
                 'ptp': 1.0,
                 'recall': 1.0,
                 'rtp': 1.0},
 'mention_ceaf': {'fn': 0.0,
                  'fp': 0.0,
                  'fscore': 1.0,
                  'precision': 1.0,
                  'ptp': 2.0,
                  'recall': 1.0,
                  'rtp': 2.0},
 'pairwise': {'fn': 0,
              'fp': 0,
              'fscore': 1.0,
              'precision': 1.0,
              'ptp': 2,
              'recall': 1.0,
              'rtp': 2},
}

def test_conll_sysa():
    assert check_correct(EXPECTED_CONLL_SYSA,
                         _get_stats(CONLL_GOLD_UNSTITCHED,
                                    CONLL_SYSA_UNSTITCHED))


EXPECTED_CONLL_MULTI_SELFEVAL = {
 'entity_match': {'fn': 0,
                       'fp': 0,
                       'fscore': 1.0,
                       'precision': 1.0,
                       'recall': 1.0,
                       'ptp': 3,
                       'rtp': 3},
 'strong_link_match': {'fn': 0,
                       'fp': 0,
                       'fscore': 1.0,
                       'precision': 1.0,
                       'recall': 1.0,
                       'ptp': 3,
                       'rtp': 3},
 'strong_nil_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 1,
                      'rtp': 1},
 'strong_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 4,
                      'rtp': 4},
 'strong_typed_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 4,
                      'rtp': 4},
 'strong_mention_match': {'fn': 0,
                          'fp': 0,
                          'fscore': 1.0,
                          'precision': 1.0,
                          'recall': 1.0,
                          'ptp': 4,
                          'rtp': 4},
 'strong_linked_mention_match': {'fn': 0,
                                 'fp': 0,
                                 'fscore': 1.0,
                                 'precision': 1.0,
                                 'recall': 1.0,
                                 'ptp': 3,
                                 'rtp': 3},
 'entity_ceaf': {'fn': 0.0,
                 'fp': 0.0,
                 'fscore': 1.0,
                 'precision': 1.0,
                 'ptp': 2.0,
                 'recall': 1.0,
                 'rtp': 2.0},
 'mention_ceaf': {'fn': 0.0,
                  'fp': 0.0,
                  'fscore': 1.0,
                  'precision': 1.0,
                  'ptp': 4.0,
                  'recall': 1.0,
                  'rtp': 4.0},
 'pairwise': {'fn': 0,
              'fp': 0,
              'fscore': 1.0,
              'precision': 1.0,
              'ptp': 4,
              'recall': 1.0,
              'rtp': 4},
}

def test_conll_multi_selfeval():
    assert check_correct(EXPECTED_CONLL_MULTI_SELFEVAL,
                         _get_stats(CONLL_MULTI_GOLD_UNSTITCHED,
                                    CONLL_MULTI_GOLD_UNSTITCHED))

EXPECTED_CONLL_MULTI_SYSA = {
 'entity_match': {'fn': 2,
                       'fp': 2,
                       'fscore': 0.3333333333333333,
                       'precision': 0.3333333333333333,
                       'recall': 0.3333333333333333,
                       'ptp': 1,
                       'rtp': 1},
 'strong_link_match': {'fn': 2,
                       'fp': 2,
                       'fscore': 0.3333333333333333,
                       'precision': 0.3333333333333333,
                       'recall': 0.3333333333333333,
                       'ptp': 1,
                       'rtp': 1},
 'strong_nil_match': {'fn': 1,
                      'fp': 1,
                      'fscore': 0.0,
                      'precision': 0.0,
                      'recall': 0.0,
                      'ptp': 0,
                      'rtp': 0},
 'strong_all_match': {'fn': 3,
                      'fp': 3,
                      'fscore': 0.25,
                      'precision': 0.25,
                      'recall': 0.25,
                      'ptp': 1,
                      'rtp': 1},
 'strong_typed_all_match': {'fn': 3,
                      'fp': 3,
                      'fscore': 0.25,
                      'precision': 0.25,
                      'recall': 0.25,
                      'ptp': 1,
                      'rtp': 1},
 'strong_mention_match': {'fn': 0,
                          'fp': 0,
                          'fscore': 1.0,
                          'precision': 1.0,
                          'recall': 1.0,
                          'ptp': 4,
                          'rtp': 4},
 'strong_linked_mention_match': {'fn': 1,
                                 'fp': 1,
                                 'fscore': 0.6666666666666666,
                                 'precision': 0.6666666666666666,
                                 'recall': 0.6666666666666666,
                                 'ptp': 2,
                                 'rtp': 2},
 'entity_ceaf': {'fn': 0.0,
                 'fp': 0.0,
                 'fscore': 1.0,
                 'precision': 1.0,
                 'ptp': 2.0,
                 'recall': 1.0,
                 'rtp': 2.0},
 'mention_ceaf': {'fn': 0.0,
                  'fp': 0.0,
                  'fscore': 1.0,
                  'precision': 1.0,
                  'ptp': 4.0,
                  'recall': 1.0,
                  'rtp': 4.0},
 'pairwise': {'fn': 0,
              'fp': 0,
              'fscore': 1.0,
              'precision': 1.0,
              'ptp': 4,
              'recall': 1.0,
              'rtp': 4},
}

def test_conll_multi_sysa():
    assert check_correct(EXPECTED_CONLL_MULTI_SYSA,
                         _get_stats(CONLL_MULTI_GOLD_UNSTITCHED,
                                    CONLL_MULTI_SYSA_UNSTITCHED))
