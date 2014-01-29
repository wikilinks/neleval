
import os
from pprint import pprint

from data import Data
from evaluate import Evaluate

DIR = os.path.join(os.path.dirname(__file__))
EXAMPLES = os.path.join(DIR, 'examples')

DATA = os.path.join(EXAMPLES, 'data.txt')

def test_data():
    d = Data.read(DATA)
    assert len(d) == 1
    doc = list(d)[0]
    assert len(doc.mentions) == 2
    assert len(doc.links) == 1
    assert len(doc.nils) == 1

def _get_stats(gold_fname, sys_fname):
    gold_path = os.path.join(EXAMPLES, gold_fname)
    sys_path = os.path.join(EXAMPLES, sys_fname)
    stats = Evaluate(sys_path, gold=gold_path)()
    pprint(stats)
    return stats

CORRECT = {
  'link_entity_match': {'fn': 0,
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
 'strong_nil_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 1},
 'weak_link_match': {'fn': 0,
                     'fp': 0,
                     'fscore': 1.0,
                     'precision': 1.0,
                     'recall': 1.0,
                     'tp': 1},
 'weak_mention_match': {'fn': 0,
                        'fp': 0,
                        'fscore': 1.0,
                        'precision': 1.0,
                        'recall': 1.0,
                        'tp': 2},
 'weak_nil_match': {'fn': 0,
                    'fp': 0,
                    'fscore': 1.0,
                    'precision': 1.0,
                    'recall': 1.0,
                    'tp': 1}}

def test_correct():
    assert CORRECT == _get_stats('data.txt', 'data.txt')

ATTEMPT = {
 'link_entity_match': {'fn': 0,
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
 'strong_nil_match': {'fn': 1,
                      'fp': 0,
                      'fscore': 0.0,
                      'precision': 1.0,
                      'recall': 0.0,
                      'tp': 0},
 'weak_link_match': {'fn': 0,
                     'fp': 1,
                     'fscore': 0.6666666666666666,
                     'precision': 0.5,
                     'recall': 1.0,
                     'tp': 1},
 'weak_mention_match': {'fn': 0,
                        'fp': 0,
                        'fscore': 1.0,
                        'precision': 1.0,
                        'recall': 1.0,
                        'tp': 2},
 'weak_nil_match': {'fn': 1,
                    'fp': 0,
                    'fscore': 0.0,
                    'precision': 1.0,
                    'recall': 0.0,
                    'tp': 0},
}
def test_attempt():
    assert ATTEMPT == _get_stats('data.txt', 'data_attempt.txt')

CORRECT_MORE = {
 'link_entity_match': {'fn': 0,
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
 'strong_nil_match': {'fn': 0,
                      'fp': 0,
                      'fscore': 1.0,
                      'precision': 1.0,
                      'recall': 1.0,
                      'tp': 1},
 'weak_link_match': {'fn': 0,
                     'fp': 0,
                     'fscore': 1.0,
                     'precision': 1.0,
                     'recall': 1.0,
                     'tp': 3},
 'weak_mention_match': {'fn': 0,
                        'fp': 0,
                        'fscore': 1.0,
                        'precision': 1.0,
                        'recall': 1.0,
                        'tp': 4},
 'weak_nil_match': {'fn': 0,
                    'fp': 0,
                    'fscore': 1.0,
                    'precision': 1.0,
                    'recall': 1.0,
                    'tp': 1},
}

def test_more_correct():
    assert CORRECT_MORE == _get_stats('data_more.txt', 'data_more.txt')

CORRECT_ATTEMPT = {
 'link_entity_match': {'fn': 2,
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
 'strong_nil_match': {'fn': 1,
                      'fp': 1,
                      'fscore': 1.0,
                      'precision': 0.0,
                      'recall': 0.0,
                      'tp': 0},
 'weak_link_match': {'fn': 2,
                     'fp': 2,
                     'fscore': 0.3333333333333333,
                     'precision': 0.3333333333333333,
                     'recall': 0.3333333333333333,
                     'tp': 1},
 'weak_mention_match': {'fn': 0,
                        'fp': 0,
                        'fscore': 1.0,
                        'precision': 1.0,
                        'recall': 1.0,
                        'tp': 4},
 'weak_nil_match': {'fn': 1,
                    'fp': 1,
                    'fscore': 1.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'tp': 0},
}


def test_more_attempt():
    assert CORRECT_ATTEMPT == _get_stats('data_more.txt', 'data_more_attempt.txt')
