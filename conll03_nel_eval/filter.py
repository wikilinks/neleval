from data import Mention, Document, Data
from utils import log

class Filter(object):
    def __init__(self, fname, split=None):
        self.fname = fname
        self.split = split

    def __call__():
        d = Data.read(self.fname)
        log('Read {} documents from {}'.format(len(d), self.fname))
        output = []
        for doc in d:
            if self.split and self.split != doc.split:
                continue
            output.append(doc.to_conll())
        return '\n'.join(output)

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('filter', help='Filter dataset by tag')
        p.add_argument('-s', '--split', help='Split to filter')
        p.set_defaults(cls=cls)
        return p
