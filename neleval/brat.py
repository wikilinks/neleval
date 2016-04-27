#!/usr/bin/env python
from collections import defaultdict
from glob import glob
import os
import urllib

from .annotation import Annotation, Candidate
from .utils import normalise_link, unicode, utf8_open


EXT = 'ann'  # extension of brat annotation files
SCORE = 1.0  # default disambiguation score
WP = 'Wikipedia:'  # likely wikipedia namespace
WP_LEN = len(WP)


class PrepareBrat(object):
    "Convert brat format for evaluation"
    def __init__(self, dir, mapping=None):
        self.dir = dir  # dir containing brat .ann files
        self.mapping = self.read_mapping(mapping)

    def __call__(self):
        return u'\n'.join(unicode(a) for a in self.annotations())

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('dir', metavar='DIR',
                       help='directory containing .ann files')
        p.add_argument('-m', '--mapping', help='mapping for titles')
        p.set_defaults(cls=cls)
        return p

    def annotations(self):
        "Return list of annotation objects"
        r = BratReader(self.dir)
        for annot_id, doc_id, start, end, name, candidates in r:
            mapped = list(self.map(candidates))
            yield Annotation(doc_id, start, end, mapped)

    def map(self, candidates):
        for c in candidates:
            kb_id = normalise_link(c.id)
            if self.mapping:
                c.id = self.mapping.get(kb_id, kb_id)
            yield c

    def read_mapping(self, mapping):
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


class BratReader(object):
    def __init__(self, dir, ext=EXT, score=SCORE):
        self.dir = dir
        self.ext = ext
        self.len = len(ext)+1
        self.score = score

    def __iter__(self):
        for doc_id, fh in self.files():
            mentions, norms = self.read(fh)
            for annot_id, start, end, name, ne_type in mentions:
                candidates = list(self.candidates(annot_id, ne_type, norms))
                yield annot_id, doc_id, start, end, name, candidates

    def files(self):
        for f in glob(os.path.join(self.dir, '*.{}'.format(self.ext))):
            doc_id = os.path.basename(f)[:-self.len]
            yield doc_id, utf8_open(f)

    def read(self, fh):
        mentions = []
        normalizations = defaultdict(list)
        for l in fh:
            l = l.strip()
            if l.startswith('T'):
                # mention annotation line
                annot_id, mention, name = l.split('\t', 2)
                ne_type, start, end = mention.split(' ', 2)
                mentions.append((annot_id, start, end, name, ne_type))
            elif l.startswith('N'):
                # normalization annotation line
                norm_id, reference = l.split('\t', 1)
                _, annot_id, kb_id = reference.split(' ', 2)
                normalizations[annot_id].append(self.normalise(kb_id))
        return mentions, normalizations

    def normalise(self, kb_id):
        return self.unquote(self.rm_namespace(kb_id))

    def unquote(self, kb_id):
        if hasattr(urllib, 'parse'):
            return urllib.parse.unquote(kb_id)
        return urllib.unquote(kb_id.encode('utf8')).decode('utf8')

    def rm_namespace(self, kb_id):
        if kb_id.startswith(WP):
            return kb_id[len(WP):]
        else:
            return kb_id

    def candidates(self, annot_id, ne_type, norms):
        for kb_id in norms.get(annot_id, []):
            yield Candidate(kb_id, self.score, ne_type)
