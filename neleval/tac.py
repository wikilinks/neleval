#!/usr/bin/env python
import itertools
import operator
from collections import defaultdict
from xml.etree.cElementTree import iterparse


from .annotation import Annotation, Candidate
from .data import ENC
from .utils import normalise_link, log

# TODO add reader for TAC before 2014

# query xml element and attribute names
QUERY_ELEM = 'query'
QID_ATTR   = 'id'
DOCID_ELEM = 'docid'
START_ELEM = 'beg'
END_ELEM   = 'end'
NAME_ELEM  = 'name'

class PrepareTac(object):
    "Convert TAC output format for evaluation"
    def __init__(self, system, queries, excluded_spans=None, mapping=None):
        assert excluded_spans
        self.system = system  # TAC links file
        self.queries = queries  # TAC queries/mentions file
        self.excluded_offsets = self.read_excluded_spans(excluded_spans)
        self.mapping = self.read_mapping(mapping)

    def __call__(self):
        return u'\n'.join(unicode(a) for a in self.annotations()).encode(ENC)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE', help='link annotations')
        p.add_argument('-q', '--queries', required=True, help='mention annotations')
        p.add_argument('-x', '--excluded-spans', help='file of spans to delete mentions in')
        p.add_argument('-m', '--mapping', help='mapping for titles')
        p.set_defaults(cls=cls)
        return p

    def annotations(self):
        "Return list of annotation objects"
        n_candidates = 0
        n_excluded = 0
        n_annotations = 0
        r = TacReader(self.system, self.queries)
        excluded = self.excluded_offsets
        for qid, docid, start, end, name, candidates in r:
            # NB: start, end are strings
            if (docid, start) in excluded or (docid, end) in excluded:
                n_excluded += 1
                continue
            n_candidates += len(candidates)
            mapped = list(self.map(candidates))
            yield Annotation(docid, start, end, mapped)
            n_annotations += 1
        log.info('Read {} candidates for {} annotations (excluded {}) '
                 'from {}'.format(n_candidates, n_annotations, n_excluded,
                                  self.system))

    def map(self, candidates):
        for c in candidates:
            kbid = normalise_link(c.id)
            if self.mapping:
                c.id = self.mapping.get(kbid, kbid)
            yield c

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

    def read_excluded_spans(self, path):
        if path is None:
            return set()

        excluded = set()
        with open(path) as f:
            for l in f:
                doc_id, start, end = l.strip().split('\t')[:3]
                for i in range(int(start), int(end) + 1):
                    excluded.add((doc_id, str(i)))
        return excluded


class TacReader(object):
    def __init__(self, links_file, queries_file):
        self.links_file = links_file
        self.queries_file = queries_file

    def __iter__(self):
        cdict = self.read_candidates()
        for (docid, start, end), queries in self.grouped_queries():
            qids, _, _, _, names = zip(*queries)

            candidates = sum((cdict.pop(qid, []) for qid in qids), [])
            candidates = sorted(candidates, reverse=True)  # sort by -score
            yield qids, docid, start, end, names, candidates

        if cdict:
            raise ValueError('Remaining annotations unaligned to '
                             'queries: {}'.format(cdict))

    def read_candidates(self):
        "Return {qid: [(score, kbid, type)]} dictionary"
        d = defaultdict(list)
        for line in open(self.links_file):
            cols = line.decode(ENC).strip().split('\t')
            if len(cols) < 3:
                continue
            if cols[0] == 'query_id':
                continue
            qid, kbid, type = cols[:3]
            score = 1.0 if len(cols) <= 3 else float(cols[3])
            d[qid].append(Candidate(kbid, score, type))
        return d

    def iter_queries(self):
        "Yield (qid, docid, start, end, name) tuples"
        for event, elem in iterparse(self.queries_file):
            if elem.tag == QUERY_ELEM:
                yield self._query(elem)

    def grouped_queries(self, key=operator.itemgetter(slice(1, 4))):
        # Deduplicate by span!
        return itertools.groupby(sorted(self.iter_queries(), key=key), key)

    def _query(self, query_elem):
        "Return (qid, docid, start, end, name) tuple"
        qid = query_elem.get(QID_ATTR)
        d = {}
        for child in query_elem:
            d[child.tag] = child.text
        return qid, d[DOCID_ELEM], d[START_ELEM], d[END_ELEM], d[NAME_ELEM]
