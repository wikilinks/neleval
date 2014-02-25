from .data import Reader, ENC

class Release(object):
    def __init__(self, fname):
        """
        fname - system output
        """
        self.fname = fname

    def __call__(self):
        lines = []
        for doc in list(sorted(Reader(open(self.fname)))):
            for m in doc.iter_mentions():
                lines.append(u'{}\t{}\t{}\t{}\t{}'.format(doc.doc_id, m.start, m.end, m.link or '', m.score or ''))
        return '\n'.join(lines).encode(ENC)
            
    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('release', help='Produce release file from system output')
        p.add_argument('fname', metavar='FILE')
        p.set_defaults(cls=cls)
        return p
