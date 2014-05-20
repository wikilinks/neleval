#!/usr/bin/env python
from .annotation import Annotation, AnnotationReader, Candidate
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
DATA = os.path.join(EXAMPLES, 'data.txt')
DATA_FULL = os.path.join(EXAMPLES, 'data_full.txt')
TAC_GOLD_QUERIES = os.path.join(EXAMPLES, 'tac_gold.xml')
TAC_GOLD_LINKS = os.path.join(EXAMPLES, 'tac_gold.tab')
TAC_GOLD_COMB = os.path.join(EXAMPLES, 'tac_gold.combined.tsv')
#TAC_SYS_QUERIES = os.path.join(EXAMPLES, 'tac_system.xml')
#TAC_SYS_LINKS = os.path.join(EXAMPLES, 'tac_system.tab')

EXPECTED_ANNOTS = [
    Annotation("bolt-eng-DF-200-192451-5799099", 2450, 2454,
               candidates=[Candidate("kb_A", 1.0, "GPE")]),
    Annotation("bolt-eng-DF-200-192453-5806828", 3287, 3295,
               candidates=[Candidate("kb_A", 1.0, "GPE")]),
    Annotation("XIN_ENG_20100703.0052", 500, 502,
               candidates=[Candidate("kb_A", 1.0, "GPE")]),
    Annotation("AFP_ENG_20100314.0466", 247, 256,
               candidates=[Candidate("kb_A", 1.0, "GPE")]),
    Annotation("AFP_ENG_20100212.0648", 338, 367,
               candidates=[Candidate("kb_B", 1.0, "PER")]),
    Annotation("APW_ENG_20100209.1003", 573, 577,
               candidates=[Candidate("kb_B", 1.0, "PER")]),
    Annotation("eng-NG-31-100506-10870984", 322, 335,
               candidates=[Candidate("NIL000", 1.0, "PER")]),
    Annotation("bolt-eng-DF-170-181103-8888234", 322, 334,
               candidates=[Candidate("NIL000", 1.0, "PER")]),
    Annotation("bolt-eng-DF-199-192909-6666623", 128269, 128275,
               candidates=[Candidate("NIL001", 1.0, "ORG")]),
    Annotation("AFP_ENG_20100120.0809", 109, 121,
               candidates=[Candidate("NIL001", 1.0, "ORG")]),
    ]
def test_annotation_reader():
    annots = list(AnnotationReader(TAC_GOLD_COMB))
    assert [str(a) for a in annots] == [str(a) for a in EXPECTED_ANNOTS]

def test_tac_prepare():
    combined = PrepareTac(TAC_GOLD_LINKS, TAC_GOLD_QUERIES)()
    assert combined == open(TAC_GOLD_COMB).read().rstrip('\n')

"""
def test_tac_eval():
    def combine(links_fname, queries_fname):
        c_fname = '{}.combined.tmp'.format(links_fname)
        with open(c_fname, 'w') as c_fh):
            print >>c_fh, PrepareTac(links, queries)()
        return c_fname
    gold = combine(TAC_GOLD_LINKS, TAC_GOLD_QUERIES)
    system = combine(TAC_SYSTEM_LINKS, TAC_SYSTEM_QUERIES)
    stats = Evaluate(gold, system, fmt='no_format')()
    check_correct(TAC_CORRECT, stats)
    os.remove(gold)
    os.remove(system)
"""

def test_normalisation():
    assert normalise_link('Some title') == 'Some_title'
    assert normalise_link('http://en.wikipedia.org/wiki/Some title') == 'Some_title'
    assert normalise_link('http://fr.wikipedia.org/wiki/Some') == 'Some'
    assert normalise_link('Some') == 'Some'

def test_data():
    d = list(Reader(open(DATA)))
    assert len(d) == 1
    doc = list(d)[0]
    assert len(list(doc.iter_mentions())) == 2
    assert len(list(doc.iter_links())) == 1
    assert len(list(doc.iter_nils())) == 1

def test_read_write():
    for f in (DATA, DATA_FULL):
        out = BytesIO()
        w = Writer(out)
        for doc in list(Reader(open(f))):
            w.write(doc)
        w_str = '\n'.join(l.rstrip('\t') for l in out.getvalue().split('\n'))
        assert w_str == open(f).read()

def test_unstitch_stitch():
    def unstitch(f):
        u_fname = '{}.unstitched.tmp'.format(f)
        with open(u_fname, 'w') as u_fh:
            print >>u_fh, Unstitch(f)()
        return u_fname
    for f in (DATA, DATA_FULL):
        u_fname = unstitch(f)
        s_str = Stitch(u_fname, f)()
        s_str = '\n'.join(l.rstrip('\t') for l in s_str.split('\n'))
        assert s_str == open(f).read()
        os.remove(u_fname)

def test_sentences():
    """ Checks that we can read contiguous sentences with the indices making sense. """
    docs = list(Reader(open(DATA_FULL)))
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

def _get_stats(gold_fname, sys_fname):
    gold_path = os.path.join(EXAMPLES, gold_fname)
    sys_path = os.path.join(EXAMPLES, sys_fname)
    stats = Evaluate(sys_path, gold=gold_path, fmt='no_format')()
    pprint(stats)
    return stats

def check_correct(expected, actual):
    assert expected.viewkeys() == actual.viewkeys(), 'Different keys\nexpected\t{}\nactual\t{}'.format(sorted(expected.keys()), sorted(actual.keys()))
    for k in expected:
        assert expected[k] == actual[k], 'Different on key "{}".\nexpected\t{}\nactual\t{}'.format(k, expected[k], actual[k])
    return True

CORRECT = {
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

def test_correct():
    assert check_correct(CORRECT, _get_stats('data.txt', 'data.txt'))

ATTEMPT = {
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

def test_attempt():
    assert check_correct(ATTEMPT, _get_stats('data.txt', 'data_attempt.txt'))

CORRECT_MORE = {
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

def test_more_correct():
    assert check_correct(CORRECT_MORE, _get_stats('data_more.txt', 'data_more.txt'))

CORRECT_ATTEMPT = {
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

def test_more_attempt():
    assert check_correct(CORRECT_ATTEMPT, _get_stats('data_more.txt', 'data_more_attempt.txt'))
