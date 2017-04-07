#!/usr/bin/env python
import itertools
import operator
import argparse
import re
from collections import defaultdict
from xml.etree.cElementTree import iterparse

from .annotation import Annotation, Candidate
from .utils import normalise_link, log, unicode, utf8_open


# TODO add reader for TAC before 2014

# query xml element and attribute names
QUERY_ELEM = 'query'
QID_ATTR   = 'id'
DOCID_ELEM = 'docid'
START_ELEM = 'beg'
END_ELEM   = 'end'
NAME_ELEM  = 'name'

class PrepareTac(object):
    """Convert TAC output format for evaluation

    queries file looks like:

        <?xml version="1.0" encoding="UTF-8"?>
        <kbpentlink>
          <query id="doc_01">
            <name>China</name>
            <docid>bolt-eng-DF-200-192451-5799099</docid>
            <beg>2450</beg>
            <end>2454</end>
          </query>
        </kbpentlink>

    links file looks like:

        doc_01	kb_A	GPE	0.95
    """
    def __init__(self, system, queries, excluded_spans=None, mapping=None):
        self.system = system  # TAC links file
        self.queries = queries  # TAC queries/mentions file
        self.excluded_offsets = read_excluded_spans(excluded_spans)
        self.mapping = read_mapping(mapping)

    def __call__(self):
        return u'\n'.join(unicode(a) for a in self.annotations())

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
            if not candidates:
                raise ValueError('No candidates found for query ' + str(qid))
            n_candidates += len(candidates)
            mapped = list(apply_mapping(self.mapping, candidates))
            yield Annotation(docid, start, end, mapped)
            n_annotations += 1
        log.info('Read {} candidates for {} annotations (excluded {}) '
                 'from {}'.format(n_candidates, n_annotations, n_excluded,
                                  self.system))


def read_mapping(mapping):
    if not mapping:
        return None
    redirects = {}
    with utf8_open(mapping) as f:
        for l in f:
            bits = l.rstrip().split('\t')
            title = bits[0].replace(' ', '_')
            for r in bits[1:]:
                r = r.replace(' ', '_')
                redirects[r] = title
            redirects[title] = title
    return redirects


def apply_mapping(mapping, candidates):
    for c in candidates:
        kbid = normalise_link(c.eid)
        if mapping:
            c.eid = mapping.get(kbid, kbid)
        yield c


def read_excluded_spans(path):
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
        for line in utf8_open(self.links_file):
            cols = line.strip().split('\t')
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


class PrepareTac15(object):
    """Convert TAC 2015 KBP EL output format for evaluation

    Format is single tab-delimited file of fields:

        * system run ID (ignored)
        * mention ID (ignored)
        * mention text (ignored)
        * offset in format "<doc ID>: <start> - <end>"
        * link (KB ID beginning "E" or "NIL")
        * entity type of {GPE, ORG, PER, LOC, FAC}
        * mention type of {NAM, NOM}
        * confidence score in (0.0, 1.0]
        * web search (ignored)
        * wiki text (ignored)
        * unknown (ignored)
    """
    def __init__(self, system, excluded_spans=None, mapping=None):
        self.system = system
        self.excluded_offsets = read_excluded_spans(excluded_spans)
        self.mapping = read_mapping(mapping)

    def __call__(self):
        return u'\n'.join(unicode(a) for a in self.read_annotations(self.system))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', metavar='FILE', type=argparse.FileType('r'), help='link annotations')
        p.add_argument('-x', '--excluded-spans', help='file of spans to delete mentions in')
        p.add_argument('-m', '--mapping', help='mapping of KB IDs to titles')
        p.set_defaults(cls=cls)
        return p

    @staticmethod
    def _read_tab_delim(f):
        for l in f:
            yield l.rstrip('\n\r').split('\t')

    def read_annotations(self, f):
        "Return list of annotation objects"
        key_fn = operator.itemgetter(3)
        grouped = itertools.groupby(sorted(self._read_tab_delim(f),
                                           key=key_fn), key_fn)
        excluded = self.excluded_offsets
        n_candidates = 0
        n_annotations = 0
        n_excluded = 0
        for key, cand_data in grouped:
            docid, start, end = self.KEY_RE.match(key).groups()
            if (docid, start) in excluded or (docid, end) in excluded:
                n_excluded += 1
                continue
            # order by descending score, i.e. column 8 of the TAC 2015 
            # KBP EL output format
            cand_data = sorted(cand_data, key=lambda x: -float(x[7]))
            candidates = []
            for cand in cand_data:
                kbid, ne_type, mention_type, score = cand[4:8]
                type = '{}/{}'.format(ne_type, mention_type)
                candidates.append(Candidate(kbid, score, type))
            mapped = list(apply_mapping(self.mapping, candidates))
            yield Annotation(docid, start, end, mapped)
            n_annotations += 1
            n_candidates += len(mapped)
        log.info('Read {} candidates for {} annotations (excluded {}) '
                 'from {}'.format(n_candidates, n_annotations, n_excluded,
                                  self.system.name))

    KEY_RE = re.compile(u'^(\\S+): ?(\\d+) ?[-\u2013] ?(\\d+)$')
