from __future__ import print_function, division, absolute_import
from collections import Counter
from collections import namedtuple
from collections import defaultdict
from argparse import FileType
import operator
import re

from .document import ENC
from .document import Reader
from .evaluate import get_measure, Evaluate
from .utils import log
from .coref_adjust import fix_unaligned


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


class AnalyzeEntities(object):
    """Analyze best-matched sys-gold entity pairs
    """

    def __init__(self, system, gold, assignment_measure='mention_ceaf_plus', counters=()):
        self.system = list(Reader(open(system)))
        self.gold = list(Reader(open(gold)))
        self.assignment_measure = get_measure(assignment_measure)
        self.counters = counters

    COLUMNS = ['true_eid', 'pred_eid',
               'tp', 'fp', 'fn',
               'doc_tp', 'doc_fp', 'doc_fn',]

    def __call__(self):
        # TODO: allow other formats
        columns = list(self.COLUMNS)
        for name, _, _ in self.counters:
            columns.append('true_' + name)
            columns.append('pred_' + name)
        print(*columns, sep='\t')

        true_all_annots = [a for doc in self.gold for a in doc.annotations]
        pred_all_annots = [a for doc in self.system for a in doc.annotations]
        true_clusters = self.assignment_measure.build_clusters(true_all_annots)
        pred_clusters = self.assignment_measure.build_clusters(pred_all_annots)
        _, _, _, _, mapping = self.assignment_measure.count_clustering(pred_clusters, true_clusters, return_mapping=True)

        true_annot_lookup = defaultdict(list)
        for a in true_all_annots:
            true_annot_lookup[a.eid].append(a)
        pred_annot_lookup = defaultdict(list)
        for a in pred_all_annots:
            pred_annot_lookup[a.eid].append(a)

        pairs = [(true_eid, pred_eid, true_clusters.pop(true_eid), pred_clusters.pop(pred_eid))
                 for true_eid, pred_eid in mapping]
        pairs.extend((true_eid, None, true_vals, set())
                     for true_eid, true_vals in true_clusters.items())
        pairs.extend((None, pred_eid, set(), pred_vals)
                     for pred_eid, pred_vals in pred_clusters.items())
        for true_eid, pred_eid, true_vals, pred_vals in pairs:
            analysis = self._analyze_entity(true_vals, pred_vals, true_annot_lookup[true_eid], pred_annot_lookup[pred_eid])
            stats, doc_stats, true_counters, pred_counters = analysis
            columns = list(stats + doc_stats)
            for true_c, pred_c in zip(true_counters, pred_counters):
                label = ';'.join('{}:{}'.format(k, v) for k, v in true_c.most_common())
                columns.append(label)
                label = ';'.join('{}:{}'.format(k, v) for k, v in pred_c.most_common())
                columns.append(label)
            print(true_eid, pred_eid, *columns, sep='\t')

    @staticmethod
    def _stats(true, pred):
        tp = len(true.intersection(pred))
        fp = len(pred.difference(true))
        fn = len(true.difference(pred))
        return tp, fp, fn

    def _analyze_entity(self, true, pred, true_anns, pred_anns):
        true_docs = set()
        true_counters = [Counter() for c in self.counters]
        for a in true_anns:
            for c, (name, f, r) in zip(true_counters, self.counters):
                c[r.search(f(a)).group()] += 1
            true_docs.add(a.docid)

        pred_docs = set()
        pred_counters = [Counter() for c in self.counters]
        for a in pred_anns:
            for c, (name, f, r) in zip(pred_counters, self.counters):
                c[r.search(f(a)).group()] += 1
            pred_docs.add(a.docid)

        return self._stats(true, pred), self._stats(true_docs, pred_docs), true_counters, pred_counters

    @staticmethod
    def _parse_counter_spec(s):
        name, s = s.split('=', 1)
        attr, pat = s.split(',', 1)
        return name, operator.attrgetter(attr), re.compile(pat)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE')
        p.add_argument('-g', '--gold', required=True)
        p.add_argument('-m', '--assignment-measure', default='mention_ceaf_plus',
                       help='measure used to map system to gold entities')
        p.add_argument('--counter', type=cls._parse_counter_spec,
                       action='append', dest='counters', default=[],
                       help='e.g. Type=type,.* will count over type fields')
        p.set_defaults(cls=cls)
        return p


class FixSpans(object):
    """Adjust unaligned mention spans

    Also: document-level evaluation
    """
    # TODO: support additional matching

    def __init__(self, system_in, system_out, gold_out=None, gold=None,
                 candidature='mention-overlap', method='max-assignment',
                 diff_out=None):

        self.system = list(Reader(FileType('r')(system_in)))
        self.gold_out = gold_out
        if gold_out is None and candidature == 'by-doc':
            raise ValueError('Fixing by doc alters the gold standard. '
                             'Please provide --gold-out')
        self.system_out = system_out
        self.gold = list(Reader(FileType('r')(gold)))
        self.diff_out = diff_out
        self.candidature = candidature
        self.method = method

    def __call__(self):
        true, pred, candidates, gold, system = self.build_candidates()

        fixes = fix_unaligned(true, pred, candidates, method=self.method)
        if self.method == 'summary':
            return

        if self.diff_out is not None:
            diff_file = FileType('w')(self.diff_out)
        else:
            diff_file = None

        for method, g_m, s_m in fixes:
            if diff_file is not None:
                print('-', method, s_m, file=diff_file)
            s_m.span = g_m.span
            if diff_file is not None:
                print('+', method, s_m, file=diff_file)

        if self.gold_out:
            with open(self.gold_out, 'w') as fout:
                for ann in gold:
                    print(ann, file=fout)
        with open(self.system_out, 'w') as fout:
            for ann in system:
                print(ann, file=fout)

    def build_candidates(self):
        name = '_build_candidates_' + self.candidature.replace('-', '_')
        try:
            func = getattr(self, name)
        except AttributeError:
            raise ValueError('Unknown candidature method: %s' %
                             self.candidature)
        return func()

    def _build_candidates_mention_overlap(self):
        measure = get_measure('strong_mention_match')

        candidates = []
        for sys_doc, gold_doc in Evaluate.iter_pairs(self.system,
                                                     self.gold):
            assert sys_doc.id == gold_doc.id
            tp, fp, fn = measure.get_matches(sys_doc.annotations,
                                             gold_doc.annotations)
            # Ensure determinism
            fp.sort()
            fn.sort()

            for g_m, _ in fn:
                for _, s_m in fp:
                    if g_m.compare_spans(s_m) in ('nested', 'crossing'):
                        candidates.append(((g_m, g_m.eid), (s_m, s_m.eid)))
                    elif s_m > g_m:
                        break
        true_clustering = measure.build_clusters([a for doc in self.gold
                                                  for a in doc.annotations])
        pred_clustering = measure.build_clusters([a for doc in self.system
                                                  for a in doc.annotations])
        return true_clustering, pred_clustering, candidates, self.gold, self.system

    def _build_candidates_by_doc(self):
        new_system = []
        new_gold = []
        measure = get_measure('strong_mention_match')
        candidates = []
        true_clustering = defaultdict(set)
        pred_clustering = defaultdict(set)
        for sys_doc, gold_doc in Evaluate.iter_pairs(self.system,
                                                     self.gold):
            tp, _, _ = measure.get_matches(sys_doc.annotations,
                                           gold_doc.annotations)
            eid_pairs = {(g_m.eid, s_m.eid) for g_m, s_m in tp}
            g_ms = {g_m.eid: g_m for g_m in gold_doc.annotations[::-1]}
            s_ms = {s_m.eid: s_m for s_m in sys_doc.annotations[::-1]}

            # Remove fn and fp that correspond to aligned clusters
            for eid, m in g_ms.items():
                true_clustering[eid].add(m)
            for eid, m in s_ms.items():
                pred_clustering[eid].add(m)

            for g_eid, s_eid in eid_pairs:
                candidates.append(((g_ms[g_eid], g_eid), (s_ms[s_eid], s_eid)))

            g_ms = set(g_ms.values())
            s_ms = set(s_ms.values())
            new_gold.extend(g_m for g_m in gold_doc.annotations
                            if g_m in g_ms)
            new_system.extend(s_m for s_m in sys_doc.annotations
                              if s_m in s_ms)

        return true_clustering, pred_clustering, candidates, new_gold, new_system

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system_in', metavar='FILE')
        p.add_argument('-o', '--system-out', metavar='FILE',
                       help='Path to write fixed annotations', required=True)
        p.add_argument('-G', '--gold-out', metavar='FILE',
                       help='Path to write adjusted gold annotations (for --by-doc)')
        p.add_argument('-d', '--diff-out', help='Path to write diff of fixes')
        p.add_argument('-g', '--gold', required=True)

        meg = p.add_mutually_exclusive_group()
        meg.add_argument('--max-assignment', dest='method', action='store_const',
                         const='max-assignment', default='max-assignment')
        meg.add_argument('--single-best', dest='method', action='store_const',
                         const='single-best')
        meg.add_argument('--unambiguous', dest='method', action='store_const',
                         const='unambiguous')
        meg.add_argument('--summary', dest='method', action='store_const',
                         const='summary')

        meg = p.add_mutually_exclusive_group()

        meg.add_argument('--by-overlap', dest='candidature',
                         const='mention-overlap', default='mention-overlap',
                         action='store_const',
                         help='Mention-level with candidates overlapping')
        meg.add_argument('--by-doc', dest='candidature', const='by-doc',
                         action='store_const',
                         help='Doc-level task candidates phrase-aligned')
        p.set_defaults(cls=cls)
        return p
