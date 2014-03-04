#!/usr/bin/env python
"""
Fetch redirects for entities in in AIDA/CoNLL-formatted file.
"""
from io import BytesIO
import re

from .data import Reader
from .wikipedia import Wikipedia

class FetchMapping(object):
    def __init__(self, fname, keep=None):
        self.fname = fname # aida/conll gold file
        self.keep = re.compile(keep) if keep else None # e.g., .*testb.*
        self.w = Wikipedia()
        self.seen = set()

    def __call__(self):
        self.data = list(Reader(open(self.fname)))
        self.redirects = dict(self.fetch())        
        out = BytesIO()
        for e, r in sorted(self.redirects.iteritems()):
            line = '\t'.join([e] + r)
            print >>out, line.encode('utf8')
        return out.getvalue()

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('fetch-mapping', help='Fetch ID mapping from Wikipedia API redirects')
        p.add_argument('fname', metavar='FILE')
        p.add_argument('-k', '--keep', help='regex pattern to capture')
        p.set_defaults(cls=cls)
        return p

    def fetch(self):
        for doc in self.data:
            if self.keep and not self.keep.match(doc.doc_id):
                continue
            for e in doc.iter_entities():
                e = ' '.join(e.split('_')) # TODO why url piece not title?
                if e in self.seen:
                    continue
                self.seen.add(e)
                current = self.w.redirected(e) # current title
                redirects = self.w.redirects(current) # all incoming redirects
                redirects = list(redirects)
                yield current, sorted(redirects)
