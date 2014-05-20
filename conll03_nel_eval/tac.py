#!/usr/bin/env python
from .data import ENC
from collections import defaultdict
from xml.etree.cElementTree import iterparse

# query xml element and attribute names
QUERY_ELEM = 'query'
QID_ATTR   = 'id'
DOCID_ELEM = 'docid'
START_ELEM = 'beg'
END_ELEM   = 'end'
NAME_ELEM  = 'name'

class PrepareTac(object):
    "Convert TAC output format for evaluation"
    def __init__(self, system, queries):
        self.system = system # TAC links file
        self.queries = queries # TAC queries/mentions file

    def __call__(self):
        return '\n'.join(str(a) for a in self.annotations()).encode(ENC)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE', help='link annotations')
        p.add_argument('-q', '--queries', help='mention annotations')
        p.set_defaults(cls=cls)
        return p

    def annotations(self):
        "Return list of annotation objects"
        r = TacReader(self.system, self.queries)
        for qid, docid, start, end, name, candidates in r:
            yield Annotation(docid, start, end, candidates)

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
            cols = line.strip().split('\t')
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

class Annotation(object):
    def __init__(self, docid, start, end, candidates=[]):
        self.docid = docid
        self.start = start
        self.end = end
        self.candidates = candidates

    def __str__(self):
        return u'{}\t{}\t{}\t{}'.format(
            self.docid,
            self.start,
            self.end,
            u'\t'.join([str(c) for c in self.candidates])
            )

class Candidate(object):
    def __init__(self, kbid, score=None, type=None):
        self.kbid = kbid
        self.score = score
        self.type = type

    def __cmp__(self, other):
        assert isinstance(other, Candidate)
        return cmp(self.score, other.score)

    def __str__(self):
        return u'{}\t{}\t{}'.format(self.kbid, self.score, self.type or '')
