#!/usr/bin/env python
"""
Fetch redirects for entities in in AIDA/CoNLL-formatted file.
"""
from io import BytesIO
import re

from .document import Reader
from .utils import normalise_link
from .wikipedia import Wikipedia

class FetchMapping(object):
    'Fetch ID mapping from Wikipedia API redirects'

    def __init__(self, fname):
        self.fname = fname # aida/conll gold file
        self.w = Wikipedia()
        self.seen = set()

    def __call__(self):
        self.data = list(Reader(open(self.fname)))
        self.redirects = dict(self.fetch())        
        out = BytesIO()
        for e, r in sorted(self.redirects.iteritems()):
            line = u'\t'.join([e] + r)
            print >>out, line.encode('utf8')
        return out.getvalue()

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('fname', metavar='FILE')
        p.set_defaults(cls=cls)
        return p

    def fetch(self):
        for doc in self.data:
            for e in doc.iter_entities():
                e = normalise_link(e)
                if e in self.seen:
                    continue
                self.seen.add(e)
                current = self.w.redirected(e) # current title
                redirects = self.w.redirects(current) # all incoming redirects
                redirects = list(redirects)
                yield current, sorted(redirects)
