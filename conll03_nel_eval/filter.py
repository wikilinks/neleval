from data import Data
from utils import log

class Filter(object):
    def __init__(self, fname, split=None, mapping=None):
        self.fname = fname
        self.split = split
        self.mapping = self.read_mapping(mapping)

    def __call__(self):
        d = Data.read(self.fname)
        log('Read {} documents from {}'.format(len(d), self.fname))
        output = []
        for doc in d:
            if self.split and self.split != doc.split:
                continue
            output.append(doc.to_conll(self.mapping))
        return '\n'.join(output)

    def read_mapping(self, mapping):
        if not mapping:
            return None
        redirects = {}
        with open(mapping) as f:
            for l in f:
                bits = l.decode('utf8').rstrip().split('\t')
                title = bits[0]
                for r in bits[1:]:
                    redirects[r] = title
        return redirects

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('filter', help='Filter dataset by tag')
        p.add_argument('fname', metavar='FILE')
        p.add_argument('-s', '--split', help='Split to filter')
        p.add_argument('-m', '--mapping', help='Mapping for titles')
        p.set_defaults(cls=cls)
        return p
