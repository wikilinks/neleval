#!/usr/bin/env python
from io import BytesIO
from pprint import pprint
import os
import warnings
from contextlib import contextmanager

from nose.tools import assert_almost_equal

from .document import Reader as AnnotationReader, Document
from .data import Reader, Mention, Writer
from .configs import ALL_MEASURES
from .evaluate import Evaluate
from .formats import Unstitch, Stitch
from .tac import PrepareTac
from .utils import normalise_link, utf8_open
from .annotation import Measure, Annotation

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

@contextmanager
def set_validation(val):
    prev_value = Document.VALIDATION
    Document.VALIDATION = val
    yield
    Document.VALIDATION = prev_value

def test_annotation_validation():
    # FIXME!!!: Update to use start-end not start-stop
    def _ex(start, stop, kbid='foo'):
        return b'docid\t{}\t{}\t{}\t1.0\tTYP'.format(start, stop, kbid)

    def _make_file(*examples):
        return BytesIO(b''.join(ex + b'\n' for ex in examples))
    duplicate_file = _make_file(_ex(0, 1), _ex(2, 3), _ex(3, 4), _ex(2, 3))

    with set_validation({'duplicate': 'ignore'}):
        reader = AnnotationReader(duplicate_file)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            list(reader)
            assert len(w) == 0, 'Expected no warning for duplicate annotations'

    duplicate_file.seek(0)
    with set_validation({'duplicate': 'warn'}):
        reader = AnnotationReader(duplicate_file)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            list(reader)
            assert len(w) == 1, 'Expected warning for duplicate annotations'
            assert 'duplicate' in str(w[-1].message)

    duplicate_file.seek(0)
    with set_validation({'duplicate': 'error'}):
        reader = AnnotationReader(duplicate_file)
        try:
            list(reader)
        except ValueError as exc:
            assert 'duplicate' in exc.message
            assert '\t2\t3\t' in exc.message
        else:
            assert False, 'Expected error to be raised on duplicate annotation'

    crossing_file = _make_file(_ex(0, 1), _ex(2, 4), _ex(3, 5))
    with set_validation({'crossing': 'warn'}):
        reader = AnnotationReader(crossing_file)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            list(reader)
            assert len(w) == 1, 'Expected warning for crossing annotations'
            assert 'crossing' in str(w[-1].message)

    for nested_file in [_make_file(_ex(2, 4), _ex(3, 4)),
                        _make_file(_ex(2, 4), _ex(2, 3)),
                        _make_file(_ex(2, 5), _ex(3, 4))]:
        with set_validation({'nested': 'warn'}):
            reader = AnnotationReader(nested_file)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                list(reader)
                assert len(w) == 1, 'Expected warning for crossing annotations'
                print(w[-1])
                assert 'nested' in str(w[-1].message)

    # TODO: test tricky case with nesting and crossing
    # TODO: test changing validation rules

def test_conll_data():
    d = list(Reader(utf8_open(CONLL_GOLD)))
    assert len(d) == 1
    doc = list(d)[0] 
    assert len(list(doc.iter_mentions())) == 2
    assert len(list(doc.iter_links())) == 1
    assert len(list(doc.iter_nils())) == 1

def test_conll_read_write():
    for f in (CONLL_GOLD, CONLL_MULTISENT):
        out = BytesIO()
        w = Writer(out)
        for doc in list(Reader(utf8_open(f))):
            w.write(doc)
        w_str = '\n'.join(l.rstrip('\t') for l in out.getvalue().split('\n'))
        assert w_str == open(f).read()

def test_sentences():
    """ Checks that we can read contiguous sentences with the indices making sense. """
    docs = list(Reader(utf8_open(CONLL_MULTISENT)))
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


from neleval.tests.util import check_correct

# EVALUATE TESTS

def _get_stats(gold_path, sys_path):
    stats = Evaluate(sys_path, gold=gold_path,
                     measures=ALL_MEASURES,  # TODO add test output for all
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
 'strong_typed_link_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 6,
                      'rtp': 6},
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
# 'pairwise': {'fn': 0,
#              'fp': 0,
#              'fscore': 1.0,
#              'precision': 1.0,
#              'ptp': 19,
#              'recall': 1.0,
#              'rtp': 19},
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
 'strong_typed_link_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
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
# 'pairwise': {'fn': 0,
#              'fp': 0,
#              'fscore': 1.0,
#              'precision': 1.0,
#              'ptp': 2,
#              'recall': 1.0,
#              'rtp': 2},
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
 'strong_typed_link_match': {'fn': 0,
                      'fp': 1,
                      'fscore': 0.6666666666666666,
                      'precision': 0.5,
                      'recall': 1.0,
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
# 'pairwise': {'fn': 0,
#              'fp': 0,
#              'fscore': 1.0,
#              'precision': 1.0,
#              'ptp': 2,
#              'recall': 1.0,
#              'rtp': 2},
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
 'strong_typed_link_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'ptp': 3,
                      'rtp': 3},
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
# 'pairwise': {'fn': 0,
#              'fp': 0,
#              'fscore': 1.0,
#              'precision': 1.0,
#              'ptp': 4,
#              'recall': 1.0,
#              'rtp': 4},
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
 'strong_typed_link_match': {'fn': 2,
                      'fp': 2,
                      'fscore': 0.3333333333333333,
                      'precision': 0.3333333333333333,
                      'recall': 0.3333333333333333,
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
# 'pairwise': {'fn': 0,
#              'fp': 0,
#              'fscore': 1.0,
#              'precision': 1.0,
#              'ptp': 4,
#              'recall': 1.0,
#              'rtp': 4},
}

def test_conll_multi_sysa():
    assert check_correct(EXPECTED_CONLL_MULTI_SYSA,
                         _get_stats(CONLL_MULTI_GOLD_UNSTITCHED,
                                    CONLL_MULTI_SYSA_UNSTITCHED))


def test_measure_overlap():
    def Ann(start, end):
        return Annotation('', start, end, [])
    ref = Ann(5, 14)  # 10 chars long
    ref2 = Ann(2, 3)  # 10 chars long
    assert_almost_equal(0., Measure.measure_overlap({ref: []}, 'max'))
    assert_almost_equal(0., Measure.measure_overlap({ref: []}, 'sum'))
    assert_almost_equal(.3, Measure.measure_overlap({ref: [Ann(1, 7)]}, 'max'))
    assert_almost_equal(.3, Measure.measure_overlap({ref: [Ann(1, 7)]}, 'sum'))
    assert_almost_equal(.4, Measure.measure_overlap({ref: [Ann(1, 7), Ann(11, 15)]}, 'max'))
    assert_almost_equal(.7, Measure.measure_overlap({ref: [Ann(1, 7), Ann(11, 15)]}, 'sum'))
    assert_almost_equal(.4, Measure.measure_overlap({ref: [Ann(1, 8), Ann(12, 15)]}, 'max'))
    assert_almost_equal(.7, Measure.measure_overlap({ref: [Ann(1, 8), Ann(12, 15)]}, 'sum'))
    assert_almost_equal(1., Measure.measure_overlap({ref: [Ann(5, 14)]}, 'max'))
    assert_almost_equal(1., Measure.measure_overlap({ref: [Ann(5, 14)]}, 'sum'))
    assert_almost_equal(1.4, Measure.measure_overlap({ref: [Ann(1, 8), Ann(12, 15)], ref2: [Ann(1, 8)]}, 'max'))
    assert_almost_equal(1.7, Measure.measure_overlap({ref: [Ann(1, 8), Ann(12, 15)], ref2: [Ann(1, 8)]}, 'sum'))

    # Overlapping is not officially supported, but we will test current behaviour
    assert_almost_equal(.9, Measure.measure_overlap({ref: [Ann(1, 7), Ann(6, 15)]}, 'max'))
    assert_almost_equal(1., Measure.measure_overlap({ref: [Ann(1, 7), Ann(6, 15)]}, 'sum'))
