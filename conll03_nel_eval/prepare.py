import re
from io import BytesIO

from .data import Reader, Writer
from .utils import log, normalise_link

class Prepare(object):
    """Map entity IDs to a different KB, normalises entity IDs, and select documents by doc-ID
    """
    def __init__(self, fname, keep=None, mapping=None):
        self.fname = fname
        self.keep = re.compile(keep) if keep else None
        self.mapping = self.read_mapping(mapping)

    def __call__(self):
        docs = list(Reader(open(self.fname)))
        log('Read {} documents from {}'.format(len(docs), self.fname))
        out = BytesIO()
        w = Writer(out)
        for doc in docs:
            if self.keep and not self.keep.match(doc.doc_id):
                continue
            for m in doc.iter_links():
                l = normalise_link(m.link)
                if self.mapping:
                    l = self.mapping.get(l, l)
                m.link = l
            w.write(doc)
        return out.getvalue()

    def read_mapping(self, mapping):
        if not mapping:
            return None
        redirects = {}
        with open(mapping) as f:
            for l in f:
                bits = l.decode('utf8').rstrip().split('\t')
                title = bits[0].replace(' ', '_')
                for r in bits[1:]:
                    r = r.replace(' ', '_')
                    redirects[r] = title
                redirects[title] = title
        return redirects

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('fname', metavar='FILE')
        p.add_argument('-k', '--keep', help='Regular expression pattern to select document IDs')
        p.add_argument('-m', '--mapping', help='Mapping for titles')
        p.set_defaults(cls=cls)
        return p
