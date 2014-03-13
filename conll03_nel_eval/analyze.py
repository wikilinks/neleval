from __future__ import print_function

from collections import namedtuple, Counter

from .data import Reader, ENC

class _Missing(object):
    def __str__(self):
        return "MISSING"

MISSING = _Missing()


class LinkingError(namedtuple('Error', 'doc_id text gold system')):
    @property
    def label(self):
        if self.gold is MISSING:
            return 'extra'
        if self.system is MISSING:
            return 'missing'
        if self.gold == self.system:
            return 'correct'
        if self.gold is None:
            return 'nil-as-link'
        if self.system is None:
            return 'link-as-nil'
        return 'wrong-link'

    @staticmethod
    def _str(val, pre):
        if val is MISSING:
            return u''
        elif val is None:
            return u'{}NIL'.format(pre)
        return u'{}"{}"'.format(pre, val)

    @property
    def _system_str(self):
        return self._str(self.system, 's')

    @property
    def _gold_str(self):
        return self._str(self.gold, 'g')

    def __str__(self):
        return u'{0.label}\t{0.doc_id}\tm"{0.text}"\t{0._gold_str}\t{0._system_str}'.format(self)


class Analyze(object):
    """Analyze errors"""
    def __init__(self, system, gold=None, unique=False, summary=False):
        self.system = system
        self.gold = gold
        self.unique = unique
        self.summary = summary

    def __call__(self):
        if self.unique:
            def _data():
                seen = set()
                for entry in self.iter_errors():
                    if entry in seen:
                        continue
                    seen.add(entry)
                    yield entry
        else:
            _data = self.iter_errors

        if self.summary:
            counts = Counter(error.label for error in _data())
            return '\n'.join('{1}\t{0}'.format(*tup)
                             for tup in counts.most_common())
        else:
            return u'\n'.join(unicode(error) for error in _data()).encode(ENC)

    def iter_errors(self):
        system = list(sorted(Reader(open(self.system))))
        gold = list(sorted((Reader(open(self.gold)))))
        for g, s in zip(gold, system):
            assert g.doc_id == s.doc_id
            tp, fp, fn = g.strong_mention_match(s)
            for g_m, s_m in tp:
                if g_m.link == s_m.link:
                    continue  # Correct case.
                yield LinkingError(g.doc_id, g_m.text, g_m.link, s_m.link)
            for _, m in fp:
                yield LinkingError(g.doc_id, m.text, MISSING, m.link)
            for m, _ in fn:
                yield LinkingError(g.doc_id, m.text, m.link, MISSING)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.add_argument('-u', '--unique', action='store_true', default=False,
                       help='Only consider unique errors')
        p.add_argument('-s', '--summary', action='store_true', default=False,
                       help='Output a summary rather than each instance')
        p.set_defaults(cls=cls)
        return p
