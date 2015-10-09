from __future__ import print_function, division, absolute_import
from collections import Counter
from collections import namedtuple
from collections import defaultdict
from argparse import FileType

from .document import ENC
from .document import Reader
from .evaluate import get_measure, Evaluate
from .utils import log


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
        self.system = list(Reader(open(system)))
        self.gold = list(Reader(open(gold)))
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
            return u'\n'.join(unicode(error) for error in _data()).encode(ENC)

    def iter_errors(self):
        for s, g in Evaluate.iter_pairs(self.system, self.gold):
            assert g.id == s.id
            tp, fp, fn = self.measure.get_matches(g.annotations, s.annotations)
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


class FixSpans(object):
    """Adjust unaligned mention spans

    This allows for "weak mention matching" by evaluating after fixing spans.
    It also aids in analysis of span errors.

    For each gold mention without an aligned system mention, all unaligned,
    overlapping system mentions are candidates. At most one will be aligned
    to match the gold span, by applying one of the following strategies:

    - unambiguous: only fix where there is a single overlapping mention
    - kbid: fix a mention that correctly predicts th KB ID of the gold mention
    - greedy: fix the unique mention whose predicted cluster has maximal
      overlap with the gold mention's cluster
    - greedy-any: fix an arbitrary mention (in case of tie) whose predicted
      cluster has maximal, non-zero overlap with the gold mention's cluster

    Repeated application may result in further alignments.
    """

    # XXX: Belongs in another module?
    def __init__(self, system_in, system_out, gold=None,
                 strategies=['unambiguous'], diff_out=None):
        self.system = list(Reader(FileType('r')(system_in)))
        self.system_out = system_out
        self.gold = list(Reader(FileType('r')(gold)))
        self.strategies = strategies or ['unambiguous']
        self.diff_out = diff_out

    def __call__(self):
        out_file = FileType('w')(self.system_out)
        if self.diff_out is not None:
            diff_file = FileType('w')(self.diff_out)
        else:
            diff_file = None

        measure = get_measure('strong_mention_match')
        matches = []
        for sys_doc, gold_doc in Evaluate.iter_pairs(self.system,
                                                     self.gold):
            assert sys_doc.id == gold_doc.id
            matches.append(measure.get_matches(sys_doc.annotations,
                                               gold_doc.annotations))

        for strategy in self.strategies:
            # NB: modifies self.system, matches in-place
            self.apply_strategy(matches, strategy, diff_file)

        for s_m in self.system:
            print(s_m, file=out_file)

    def apply_strategy(self, matches, strategy, diff_file=None):
        n_fixes = 0
        n_unaligned_gold = 0
        n_unaligned_sys = 0
        n_partial_match_gold = 0

        if strategy in ('greedy', 'greedy-any'):
            contingency = defaultdict(set)
            for tp, fp, fn in matches:
                for g_m, s_m in tp:
                    contingency[g_m.eid, s_m.eid].add(s_m)
            overlap_counts = {k: len(v) for k, v in contingency.items()}
            del contingency

        for tp, fp, fn in matches:
            n_unaligned_gold += len(fn)
            n_unaligned_sys += len(fp)

            if not fp or not fn:
                continue

            fixes = {}

            for g_m, _ in fn:
                candidates = []
                for _, s_m in fp:
                    if s_m in fixes:
                        continue
                    if g_m.compare_spans(s_m) in ('nested', 'crossing'):
                        candidates.append(s_m)
                if not candidates:
                    continue
                n_partial_match_gold += 1

                found = False
                if strategy == 'unambiguous':
                    if len(candidates) == 1:
                        fixes[candidates[0]] = g_m
                elif not g_m.is_nil and strategy in ('kbid', 'greedy',
                                                     'greedy-any'):
                    found = False
                    for s_m in candidates:
                        if s_m.kbid == g_m.kbid:
                            fixes[s_m] = g_m
                            found = True
                            break
                if not found and strategy in ('greedy', 'greedy-any'):
                    criterion = [overlap_counts.get((g_m.eid, s_m.eid), 0)
                                 for s_m in candidates]
                    m = max(criterion)
                    if m == 0:
                        # XXX: reconsider this case
                        # better off unaligned?
                        continue
                    s_m = candidates[criterion.index(m)]
                    if strategy in ('kbid', 'greedy-any') \
                       or criterion.count(m) == 1:
                        fixes[s_m] = g_m

            n_fixes += len(fixes)
            for s_m, g_m in fixes.items():
                if diff_file is not None:
                    print('-', s_m, file=diff_file)
                s_m.start = g_m.start
                s_m.end = g_m.end
                fp.remove((None, s_m))
                fn.remove((g_m, None))
                tp.append((g_m, s_m))
                if diff_file is not None:
                    print('+', s_m, file=diff_file)
        log.info('for strategy %s: '
                 'missing %d spurious %d overlap %d fixed %d' %
                 (strategy, n_unaligned_gold, n_unaligned_sys,
                  n_partial_match_gold, n_fixes))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system_in', metavar='FILE')
        p.add_argument('-o', '--system-out', metavar='FILE',
                       help='Path to write fixed annotations')
        p.add_argument('-d', '--diff-out', help='Path to write diff of fixes')
        p.add_argument('-g', '--gold', required=True)
        # Only "unambiguous" doesn't cheat in terms of label, but it may also
        # create more meaningless error types for analysis
        p.add_argument('-s', '--strategy', dest='strategies', action='append',
                       choices=['unambiguous', 'kbid', 'greedy', 'greedy-any'],
                       help='What approaches to apply; multiple -s flags may '
                            'be used. Default: unambiguous.')
        p.set_defaults(cls=cls)
        return p
