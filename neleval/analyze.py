from collections import Counter
from collections import namedtuple

from .document import Reader
from .evaluate import get_measure, Evaluate
from .utils import utf8_open, unicode


class _Missing(object):
    def __str__(self):
        return "MISSING"

MISSING = _Missing()


class LinkingError(namedtuple('Error', 'doc_id gold system')):
    @property
    def label(self):
        if self.gold is MISSING:
            return 'extra'
        if self.system is MISSING:
            return 'missing'
        if self.gold is None and self.system is None:
            return 'correct nil'
        if self.gold == self.system:
            return 'correct link'
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
        return u'{0.label}\t{0.doc_id}\t{0._gold_str}\t{0._system_str}'.format(self)


class Analyze(object):
    """Analyze errors"""
    def __init__(self, system, gold=None, unique=False, summary=False, with_correct=False):
        self.system = list(Reader(utf8_open(system)))
        self.gold = list(Reader(utf8_open(gold)))
        self.unique = unique
        self.summary = summary
        self.with_correct = with_correct
        self.measure = get_measure('strong_mention_match')

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
            return u'\n'.join(unicode(error) for error in _data())

    def iter_errors(self):
        for s, g in Evaluate.iter_pairs(self.system, self.gold):
            assert g.id == s.id
            tp, fp, fn = self.measure.get_matches(s.annotations, g.annotations)
            for g_m, s_m in tp:
                if g_m.kbid == s_m.kbid and not self.with_correct:
                    #continue  # Correct case.
                    yield LinkingError(g.id, g_m.kbid, s_m.kbid)
                else:
                    yield LinkingError(g.id, g_m.kbid, s_m.kbid)
            for _, m in fp:
                yield LinkingError(g.id, MISSING, m.kbid)
            for m, _ in fn:
                yield LinkingError(g.id, m.kbid, MISSING)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold', required=True)
        p.add_argument('-u', '--unique', action='store_true', default=False,
                       help='Only consider unique errors')
        p.add_argument('-s', '--summary', action='store_true', default=False,
                       help='Output a summary rather than each instance')
        p.add_argument('-c', '--with-correct', action='store_true', default=False,
                       help='Output correct entries as well as errors')
        p.set_defaults(cls=cls)
        return p
