#!/usr/bin/env python
from .document import Reader as AnnotationReader, ALL_MATCHES
from .data import Reader, Mention, Writer
from .evaluate import Evaluate
from .formats import Unstitch, Stitch
from .tac import PrepareTac
from .utils import normalise_link
from io import BytesIO
from pprint import pprint
import os

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

# EVAL TESTS

def _get_stats(gold_path, sys_path):
    stats = Evaluate(sys_path, gold=gold_path,
                     matches=ALL_MATCHES, fmt='no_format')()
    pprint(stats)
    return stats

def check_correct(expected, actual):
    assert expected.viewkeys() == actual.viewkeys(), 'Different keys\nexpected\t{}\nactual\t{}'.format(sorted(expected.keys()), sorted(actual.keys()))
    for k in expected:
        assert expected[k] == actual[k], 'Different on key "{}".\nexpected\t{}\nactual\t{}'.format(k, expected[k], actual[k])
    return True

EXPECTED_TAC_SYS = {
 'entity_match': {'fn': 0,
                  'fp': 0,
                  'fscore': 1.0,
                  'precision': 1.0,
                  'recall': 1.0,
                  'tp': 6},
 'strong_link_match': {'fn': 0,
                       'fp': 0,
                       'fscore': 1.0,
                       'precision': 1.0,
                       'recall': 1.0,
                       'tp': 6},
 'strong_nil_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 4},
 'strong_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 10},
 'strong_typed_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 10},
 'strong_mention_match': {'fn': 0,
                          'fp': 0,
                          'fscore': 1.0,
                          'precision': 1.0,
                          'recall': 1.0,
                          'tp': 10},
 'strong_linked_mention_match': {'fn': 0,
                                 'fp': 0,
                                 'fscore': 1.0,
                                 'precision': 1.0,
                                 'recall': 1.0,
                                 'tp': 6},
}

def test_tac_eval():
    check_correct(EXPECTED_TAC_SYS, _get_stats(TAC_GOLD_COMB, TAC_SYS_COMB))

EXPECTED_CONLL_SELFEVAL = {
 'entity_match': {'fn': 0,
                       'fp': 0,
                       'fscore': 1.0,
                       'precision': 1.0,
                       'recall': 1.0,
                       'tp': 1},
 'strong_link_match': {'fn': 0,
                       'fp': 0,
                       'fscore': 1.0,
                       'precision': 1.0,
                       'recall': 1.0,
                       'tp': 1},
 'strong_nil_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 1},
 'strong_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 2},
 'strong_typed_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 2},
 'strong_mention_match': {'fn': 0,
                          'fp': 0,
                          'fscore': 1.0,
                          'precision': 1.0,
                          'recall': 1.0,
                          'tp': 2},
 'strong_linked_mention_match': {'fn': 0,
                                 'fp': 0,
                                 'fscore': 1.0,
                                 'precision': 1.0,
                                 'recall': 1.0,
                                 'tp': 1},
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
                       'tp': 1},
 'strong_link_match': {'fn': 0,
                       'fp': 1,
                       'fscore': 0.6666666666666666,
                       'precision': 0.5,
                       'recall': 1.0,
                       'tp': 1},
 'strong_nil_match': {'fn': 1,
                      'fp': 0,
                      'fscore': 0.0,
                      'precision': 1.0,
                      'recall': 0.0,
                      'tp': 0},
 'strong_all_match': {'fn': 1,
                      'fp': 1,
                      'fscore': 0.5,
                      'precision': 0.5,
                      'recall': 0.5,
                      'tp': 1},
 'strong_typed_all_match': {'fn': 1,
                      'fp': 1,
                      'fscore': 0.5,
                      'precision': 0.5,
                      'recall': 0.5,
                      'tp': 1},
 'strong_mention_match': {'fn': 0,
                          'fp': 0,
                          'fscore': 1.0,
                          'precision': 1.0,
                          'recall': 1.0,
                          'tp': 2},
 'strong_linked_mention_match': {'fn': 0,
                                 'fp': 1,
                                 'fscore': 0.6666666666666666,
                                 'precision': 0.5,
                                 'recall': 1.0,
                                 'tp': 1},
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
                       'tp': 3},
 'strong_link_match': {'fn': 0,
                       'fp': 0,
                       'fscore': 1.0,
                       'precision': 1.0,
                       'recall': 1.0,
                       'tp': 3},
 'strong_nil_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 1},
 'strong_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 4},
 'strong_typed_all_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 4},
 'strong_mention_match': {'fn': 0,
                          'fp': 0,
                          'fscore': 1.0,
                          'precision': 1.0,
                          'recall': 1.0,
                          'tp': 4},
 'strong_linked_mention_match': {'fn': 0,
                                 'fp': 0,
                                 'fscore': 1.0,
                                 'precision': 1.0,
                                 'recall': 1.0,
                                 'tp': 3},
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
                       'tp': 1},
 'strong_link_match': {'fn': 2,
                       'fp': 2,
                       'fscore': 0.3333333333333333,
                       'precision': 0.3333333333333333,
                       'recall': 0.3333333333333333,
                       'tp': 1},
 'strong_nil_match': {'fn': 1,
                      'fp': 1,
                      'fscore': 1.0,
                      'precision': 0.0,
                      'recall': 0.0,
                      'tp': 0},
 'strong_all_match': {'fn': 3,
                      'fp': 3,
                      'fscore': 0.25,
                      'precision': 0.25,
                      'recall': 0.25,
                      'tp': 1},
 'strong_typed_all_match': {'fn': 3,
                      'fp': 3,
                      'fscore': 0.25,
                      'precision': 0.25,
                      'recall': 0.25,
                      'tp': 1},
 'strong_mention_match': {'fn': 0,
                          'fp': 0,
                          'fscore': 1.0,
                          'precision': 1.0,
                          'recall': 1.0,
                          'tp': 4},
 'strong_linked_mention_match': {'fn': 1,
                                 'fp': 1,
                                 'fscore': 0.6666666666666666,
                                 'precision': 0.6666666666666666,
                                 'recall': 0.6666666666666666,
                                 'tp': 2},
}

def test_conll_multi_sysa():
    assert check_correct(EXPECTED_CONLL_MULTI_SYSA,
                         _get_stats(CONLL_MULTI_GOLD_UNSTITCHED,
                                    CONLL_MULTI_SYSA_UNSTITCHED))
