import re
from data import Reader, Writer
from utils import log
from cStringIO import StringIO

class Filter(object):
    def __init__(self, fname, keep=None, mapping=None):
        self.fname = fname
        self.keep = re.compile(keep) if keep else None
        self.mapping = self.read_mapping(mapping)

    def __call__(self):
        docs = list(Reader(open(self.fname)))
        log('Read {} documents from {}'.format(len(docs), self.fname))
        out = StringIO()
        w = Writer(out)
        for doc in docs:
            if self.keep and not self.keep.match(doc.doc_id):
                continue
            if self.mapping:
                for m in doc.iter_links():
                    l = m.link.replace(' ', '_')
                    m.link = self.mapping.get(l, l)
            w.write(doc)
        return out.getvalue()

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
                redirects[title] = title
        return redirects

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('filter', help='Filter dataset by tag')
        p.add_argument('fname', metavar='FILE')
        p.add_argument('-k', '--keep', help='Regular expression pattern to capture')
        p.add_argument('-m', '--mapping', help='Mapping for titles')
        p.set_defaults(cls=cls)
        return p
