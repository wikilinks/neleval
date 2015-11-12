import sys
import argparse

from .coref_metrics import read_conll_coref
from .annotation import Annotation, Candidate


class PrepareConllCoref(object):
    "Import format from CoNLL coreference for evaluation"
    def __init__(self, input, mapping=None):
        self.input = input

    def __call__(self):
        clusters = read_conll_coref(self.input)
        return u'\n'.join(unicode(a) for a in self.annotations()).encode(ENC)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('--input', default=sys.stdin, type=argparse.FileType('r'))
        p.add_argument('--with-kb', default=False, action='store_true', help='By default all cluster labels are treated as NILs. This flag treats all as KB IDs unless prefixed by "NIL"')
        p.add_argument('--cross-doc', default=False, action='store_true', help='By default, label space is independent per document. This flag assumes global label space.')
        p.set_defaults(cls=cls)
        return p
