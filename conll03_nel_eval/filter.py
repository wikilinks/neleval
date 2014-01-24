from data import Mention, Document, Data
from utils import log

class Filter(object):
    def __init__(self, fname, split=None):
        d = Data.from_file(fname)
        log('Read {} documents from {}'.format(len(d), fname))
        for doc in d:
            if split and split != doc.split:
                continue
            print doc.to_conll()

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('filter', help='Filter dataset by tag')
        p.add_argument('-s', '--split', help='Split to filter')
        p.set_defaults(cls=cls)
        return p
