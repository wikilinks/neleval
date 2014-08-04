#!/usr/bin/env python
from .annotation import Annotation, Candidate
from .data import ENC
from .utils import normalise_link
from collections import defaultdict
from xml.etree.cElementTree import iterparse

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
    def __init__(self, system, queries, mapping=None):
        self.system = system # TAC links file
        self.queries = queries # TAC queries/mentions file
        self.mapping = self.read_mapping(mapping)

    def __call__(self):
        return u'\n'.join(unicode(a) for a in self.annotations()).encode(ENC)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE', help='link annotations')
        p.add_argument('-q', '--queries', required=True, help='mention annotations')
        p.add_argument('-m', '--mapping', help='mapping for titles')
        p.set_defaults(cls=cls)
        return p

    def annotations(self):
        "Return list of annotation objects"
        r = TacReader(self.system, self.queries)
        for qid, docid, start, end, name, candidates in r:
            mapped = list(self.map(candidates))
            yield Annotation(docid, start, end, mapped)

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

class TacReader(object):
    def __init__(self, links_file, queries_file):
        self.links_file = links_file
        self.queries_file = queries_file

    def __iter__(self):
        cdict = self.read_candidates()
        for qid, docid, start, end, name in self.iter_queries():
            candidates = sorted(cdict.get(qid, []), reverse=True)
            yield qid, docid, start, end, name, candidates

    def read_candidates(self):
        "Return {qid: [(score, kbid, type)]} dictionary"
        d = defaultdict(list)
        for line in open(self.links_file):
            cols = line.decode(ENC).strip().split('\t')
            if len(cols) < 3:
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

    def _query(self, query_elem):
        "Return (qid, docid, start, end, name) tuple"
        qid = query_elem.get(QID_ATTR)
        d = {}
        for child in query_elem:
            d[child.tag] = child.text
        return qid, d[DOCID_ELEM], d[START_ELEM], d[END_ELEM], d[NAME_ELEM]
