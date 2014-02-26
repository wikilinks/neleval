from cStringIO import StringIO
from .data import Reader, ENC, Writer

class Merge(object):
    def __init__(self, fname, gold=None):
        """
        fname - system output (release format)
        gold - gold standard
        """
        self.fname = fname
        self.gold = gold

    def __call__(self):
        # Read release file.
        data = {}
        with open(self.fname) as f:
            for l in f:
                doc_id, start, end, link, score = l.decode(ENC).rstrip('\n').split('\t')
                if not doc_id in data:
                    data[doc_id] = []
                data[doc_id].append((int(start), int(end), link or None, 
                                    float(score) if score else None))
        # Merge into docs. 
        docs = list(sorted(Reader(open(self.gold))))
        out = StringIO()
        w = Writer(out)
        for doc in docs:
            doc.clear_mentions()
            doc.set_mentions(data.get(doc.doc_id, []))
            w.write(doc)
        return out.getvalue()
            
    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('merge', help='Merge release file with gold-standard.')
        p.add_argument('fname', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.set_defaults(cls=cls)
        return p
