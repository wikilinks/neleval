
import os
from io import BytesIO
from pprint import pprint

from data import Reader, Mention, Writer
from formats import Unstitch, Stitch
from evaluate import Evaluate
from utils import normalise_link

DIR = os.path.join(os.path.dirname(__file__))
EXAMPLES = os.path.join(DIR, 'examples')

DATA = os.path.join(EXAMPLES, 'data.txt')
DATA_FULL = os.path.join(EXAMPLES, 'data_full.txt')

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
        u_fh = open(u_fname, 'w')
        u_str = Unstitch(f)()
        print >>u_fh, u_str
        u_fh.close()
        return u_fname
    for f in (DATA, DATA_FULL):
        u_fname = unstitch(f)
        s_str = Stitch(u_fname, f)()
        s_str = '\n'.join(l.rstrip('\t') for l in s_str.split('\n'))
        assert s_str == open(f).read()

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
