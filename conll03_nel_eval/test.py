
import os
from pprint import pprint
from cStringIO import StringIO

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

CORRECT = {'strong_link_match': {'fn': 0,
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
    stats = Evaluate(DATA, DATA)()
    pprint(stats)
    assert CORRECT == stats
