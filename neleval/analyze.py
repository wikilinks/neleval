from __future__ import print_function, division, absolute_import
from collections import Counter
from collections import namedtuple
from collections import defaultdict
from argparse import FileType
import copy
import heapq

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


class FixSpans(object):
    """Adjust unaligned mention spans

    This allows for "weak mention matching" by evaluating after fixing spans.
    It also aids in analysis of span errors.

    The adjustment algorithm approximately maximises the score after fixing.
    For each gold mention without an aligned system mention, all unaligned,
    overlapping system mentions are candidates. At most one s will be aligned
    to match the gold span g, according to the following rules:

        1. If, at any time, there is a 1:1 overlap confusion between g and s.
           This is called "1to1", with the "1to1-late" variant in cases where
           a confusion became 1:1 only after rule 3 was applied.
        2. Where the KB ID linked from s matches that of overlapping g.
           (s are processed in order of increasing number of candidate gs.)
        3. Greedily fix candidates that produce the largest cluster
           intersection by repeatedly selecting the (gold ID, system ID) with
           the highest true positive + confusion candidate count.

    The textual order of gold mentions is used to break ties otherwise.

    """
    # TODO: a variant that takes account of other attribs, like type

    # XXX: Belongs in another module?
    def __init__(self, system_in, system_out, gold=None,
                 use_kbid=True, use_1to1=True, use_manyto1=False,
                 use_inter=True, diff_out=None):
        self.system = list(Reader(FileType('r')(system_in)))
        self.system_out = system_out
        self.gold = list(Reader(FileType('r')(gold)))
        self.diff_out = diff_out
        self.use_kbid = use_kbid
        self.use_1to1 = use_1to1
        self.use_manyto1 = use_manyto1
        self.use_inter = use_inter

    def __call__(self):
        out_file = FileType('w')(self.system_out)
        if self.diff_out is not None:
            diff_file = FileType('w')(self.diff_out)
        else:
            diff_file = None

        self._apply(diff_file, self.use_kbid, self.use_1to1, self.use_manyto1, self.use_inter)

        for s_m in self.system:
            print(s_m, file=out_file)

    def _apply(self, diff_file, use_kbid=True, use_1to1=True,
               use_manyto1=False, use_inter=True):
        """
        1. Fix kbid matches (ordered by increasing number of overlapping golds,
           just in case!)
        2. Match one-to-one unambiguous. This cannot make more unambiguous.
        3. Group candidate fixes by cluster intersections (g.eid, s.eid),
           processed in order of decreasing true positive plus candidate
           intersection size (i.e. proposed size of intersection)
        4. Fix each cluster intersection in order. Ramifications:
             - any pairs involving g or s can be ignored
             - for all g' overlapping s, any (g', s') may now be unambiguous,
               affecting strongly connected confusion subgraph, also demoting
               affected cluster intersections in the greedy search.
        """
        n_fixes = Counter()
        measure = get_measure('strong_mention_match')

        contingency_tp_count = Counter()

        g_candidates = defaultdict(list)
        s_candidates = defaultdict(list)

        for sys_doc, gold_doc in Evaluate.iter_pairs(self.system,
                                                     self.gold):
            assert sys_doc.id == gold_doc.id
            tp, fp, fn = measure.get_matches(sys_doc.annotations,
                                             gold_doc.annotations)
            # Ensure determinism
            tp.sort()
            fp.sort()
            fn.sort()

            contingency_tp_count.update((g_m.eid, s_m.eid)
                                        for g_m, s_m in tp)

            for g_m, _ in fn:
                for _, s_m in fp:
                    if g_m.compare_spans(s_m) in ('nested', 'crossing'):
                        g_candidates[g_m].append(s_m)
                        s_candidates[s_m].append(g_m)
                    elif s_m > g_m:
                        break

        g_candidates.default_factory = None
        s_candidates.default_factory = None

        def fix(g_m, s_m, method):
            log.debug('{} fixing <{s.span}: {s.eid}> to '
                      '<{g.span}: {g.eid}>'.format(method, s=s_m, g=g_m))
            n_fixes[method] += 1
            if diff_file is not None:
                print('-', method, s_m, file=diff_file)
            s_m.start = g_m.start
            s_m.end = g_m.end
            if diff_file is not None:
                print('+', method, s_m, file=diff_file)

            try:
                cc = contingency_candidates
                requeue = []
            except NameError:
                cc = None
                requeue = None

            for s_m_other in g_candidates[g_m]:
                if s_m_other != s_m:
                    s_candidates[s_m_other].remove(g_m)
                    if cc is not None:
                        cc[g_m.eid, s_m_other.eid].remove((g_m, s_m_other))
                        requeue.append((g_m.eid, s_m_other.eid))

            for g_m_other in s_candidates[s_m]:
                if g_m_other != g_m:
                    g_candidates[g_m_other].remove(s_m)
                    if cc is not None:
                        cc[g_m_other.eid, s_m.eid].remove((g_m_other, s_m))
                        requeue.append((g_m_other.eid, s_m.eid))

            del g_candidates[g_m]
            del s_candidates[s_m]

            contingency_tp_count[g_m.eid, s_m.eid] += 1
            return requeue

        # 1. Fix kbid matches
        if use_kbid:
            kbid_matches = sorted(((s_m, [g_m for g_m in g_ms
                                          if g_m.kbid == s_m.kbid])
                                   for s_m, g_ms in s_candidates.iteritems()
                                   if s_m.is_linked),
                                  key=lambda tup: (len(tup[1]), tup[0]))
            for s_m, g_ms in kbid_matches:
                # check which are still available
                g_ms = [g_m for g_m in g_ms if g_m in g_candidates]
                if not g_ms:
                    continue
                # arbitrarily fix to match first
                fix(g_ms[0], s_m, 'kbid')

        # 2a. Fix one-to-one unambiguous
        # 2b. Fix many-to-one unambiguous: reduces precision in most metrics
        sorted_g = sorted(g_candidates)  # determinism
        if use_1to1 or use_manyto1:
            for g_m in sorted_g:
                s_ms = g_candidates[g_m]
                if len(s_ms) != 1:
                    continue
                if len(s_candidates[s_ms[0]]) == 1:
                    fix(g_m, s_ms[0], '1to1')
                elif use_manyto1:
                    g_ms = s_candidates[s_m]
                    if any(len(g_candidates[g_m_other]) != 1
                           for g_m_other in g_ms):
                        continue
                    for i, g_m_other in enumerate(g_ms):
                        if i == 0:
                            fix(g_m, s_ms[0], 'manyto1')
                        else:
                            # TODO: note reduplication in diff_out
                            # TODO: forge s_candidates, contingency_candidates
                            s_m = copy.deepcopy(s_ms[0])
                            # TODO: ensure sort order
                            self.system.append(s_m)
                            fix(g_m, s_m, 'manyto1')

        if not use_inter:
            log.info("Fixed %r" % dict(n_fixes))
            return

        # 3. Group remaining candidate fixes by cluster intersections
        contingency_candidates = defaultdict(list)
        for g_m in sorted_g:
            for s_m in g_candidates.get(g_m, []):
                contingency_candidates[g_m.eid, s_m.eid].append((g_m, s_m))

        def calc_priority(g_eid, s_eid):
            # FIXME: it's possible that the same mention is repeated in the
            #        cluster intersection, which `len` double-counts.
            return -(contingency_tp_count[g_eid, s_eid] +
                     len(contingency_candidates[g_eid, s_eid]))

        # Redundant storage of cands is for determinism
        heap = ((calc_priority(g_eid, s_eid), cands, g_eid, s_eid)
                for (g_eid, s_eid), cands
                in contingency_candidates.iteritems())
        # Ignore cases where there's no benefit to fixing
        # XXX: Perhaps have option to keep these
        heap = [tup for tup in heap if tup[0] < -1]
        heapq.heapify(heap)

        # 4. Fix each cluster intersection in order
        while heap:
            priority, cands, g_eid, s_eid = heapq.heappop(heap)
            if g_eid is None or s_eid is None:
                # Not clustered
                continue
            if priority != calc_priority(g_eid, s_eid):
                # outdated entry
                continue
            log.debug('Fixing %d candidates in cluster intersection (%s, %s) '
                      'with priority %d' %
                      (len(cands), g_eid, s_eid, priority))

            requeue = set()
            connected_cands = []
            while cands or connected_cands:
                if connected_cands:
                    method = '1to1-late'
                    g_m, s_m = connected_cands.pop()
                else:
                    method = 'inter'
                    g_m, s_m = cands.pop()

                if g_m not in g_candidates or s_m not in s_candidates:
                    # already fixed
                    continue

                affected_g = s_candidates[s_m][:]
                requeue.update(fix(g_m, s_m, method))
                # check for new unambiguous
                if use_1to1:
                    for g_m_other in affected_g:
                        if g_m_other == g_m:
                            continue
                        s_ms = g_candidates[g_m_other]
                        if len(s_ms) == 1 and len(s_candidates[s_ms[0]]) == 1:
                            connected_cands.append((g_m_other, s_ms[0]))

            for g_eid, s_eid in requeue:
                priority = calc_priority(g_eid, s_eid)
                if priority >= -1:
                    # no benefit to fixing
                    continue
                heapq.heappush(heap, (priority,
                                      contingency_candidates[g_eid, s_eid],
                                      g_eid, s_eid))

        log.info("Fixed %r" % dict(n_fixes))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system_in', metavar='FILE')
        p.add_argument('-o', '--system-out', metavar='FILE',
                       help='Path to write fixed annotations')
        p.add_argument('-d', '--diff-out', help='Path to write diff of fixes')
        p.add_argument('-g', '--gold', required=True)
        p.add_argument('--no-kbid', dest='use_kbid', action='store_false',
                       help='Turns off the heuristic of matching KB ID')
        p.add_argument('--no-1to1', dest='use_1to1', action='store_false',
                       help='Turns off the heuristic of fixing all 1-to-1 '
                            'confusions')
        p.add_argument('--use-manyto1', dest='use_1to1', action='store_true',
                       help='Activates a heuristic which reduplicates a system'
                            'span aligned to multiple gold spans.')
        p.add_argument('--no-inter', dest='use_inter', action='store_false',
                       help='Turns off the heuristic of searching for maximal '
                            'gold and system cluster interesections')
        p.set_defaults(cls=cls)
        return p
