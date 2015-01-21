#!/usr/bin/env python
from .annotation import Annotation, Candidate
from .data import ENC
from .utils import normalise_link
from collections import defaultdict
from glob import glob
import os

EXT = 'ann' # extension of brat annotation files
SCORE = 1.0 # default disambiguation score

class PrepareBrat(object):
    "Convert brat format for evaluation"
    def __init__(self, dir, mapping=None):
        self.dir = dir # dir containing brat .ann files
        self.mapping = self.read_mapping(mapping)

    def __call__(self):
        return u'\n'.join(unicode(a) for a in self.annotations()).encode(ENC)

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
        for aid, docid, start, end, name, candidates in r:
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

class BratReader(object):
    def __init__(self, dir, ext=EXT, score=SCORE):
        self.dir = dir
        self.ext = ext
        self.len = len(ext)+1
        self.score = score

    def __iter__(self):
        for docid, fh in self.files():
            mentions, resolutions = self.read(fh)
            for aid, start, end, name, type in mentions:
                candidates = [Candidate(resolutions.get(aid), self.score, type)]
                yield aid, docid, start, end, name, candidates

    def files(self):
        for f in glob(os.path.join(self.dir, '*.{}'.format(self.ext))):
            docid = os.path.basename(f)[:-self.len]
            yield docid, open(f)

    def read(self, fh):
        mentions = []
        resolutions = defaultdict(list)
        for line in fh:
            line = line.strip()
            if line.strip().startswith('#'):
                # comment value is a resolution id
                comment_id, annot_id, comment = line.split('\t')
                _, annot_id = annot_id.split()
                resolutions[annot_id] = comment
            else:
                # annotation
                annot_id, mention, name = line.split('\t')
                type, start, end = mention.split()
                mentions.append((annot_id, start, end, name, type))
        return mentions, resolutions
