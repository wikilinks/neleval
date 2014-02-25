from .data import Reader, Writer, strong_key, ENC
from utils import log
from .evaluate import Evaluate

class Analyze(object):
    def __init__(self, fname, gold=None):
        self.fname = fname
        self.gold = gold

    def __call__(self):
        lines = []
        self.system = list(sorted(Reader(open(self.fname))))
        self.gold = list(sorted((Reader(open(self.gold)))))
        for g, s in zip(self.gold, self.system):
            assert g.doc_id == s.doc_id
            index = {}
            tp, fp, fn = g.strong_mention_match(s)
            for g_m, s_m in tp:
                if g_m.link == s_m.link:
                    continue # Correct case.
                elif g_m.link and s_m.link and g_m.link != s_m.link:
                    label = 'wrong-kb'
                elif g_m.link and s_m.link is None:
                    label = 'link-as-nil'
                elif g_m.link is None and s_m.link:
                    label = 'nil-as-link'
                else:
                    assert False, 'This should not happen\tgold={}\tsystem={}'.format(g_m.link, s_m.link)
                lines.append(u'{}\t{}\tm"{}"\tg"{}"\ts"{}"'.format(label, g.doc_id, g_m.text, g_m.link, s_m.link))
            label = 'extra'
            for _, m in fp:
                lines.append(u'{}\t{}\tm"{}"\ts"{}"'.format(label, g.doc_id, m.text, m.link))
            label = 'missing'
            for m, _ in fn:
                lines.append(u'{}\t{}\tm"{}"\ts"{}"'.format(label, g.doc_id, m.text, m.link))
        return u'\n'.join(lines).encode(ENC)

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('analyze', help='Analyze errors')
        p.add_argument('fname', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.set_defaults(cls=cls)
        return p
