from data import Mention, Document

class Filter(object):
    def __init__(self, fname, split=None):
        print 'Filtering', fname, split

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('filter', help='Filter dataset by tag')
        p.add_argument('-s', '--split', help='Split to filter')
        p.set_defaults(cls=cls)
        return p
