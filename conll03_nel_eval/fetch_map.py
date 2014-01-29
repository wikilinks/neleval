#!/usr/bin/env python
"""
Fetch redirects for entities in gold standard.
"""
import sys
from data import Data
from wikipedia import Wikipedia
from utils import log

def fetch(data):
    w = Wikipedia()
    seen = set()
    for d in data.documents.values():
        for e in d.entities:
            if e in seen:
                continue
            current = w.redirected(e) # fetch current title
            redirects = w.redirects(current) # fetch all incoming redirects
            yield current, sorted(list(redirects))

def write(redirects, fh=sys.stdout):
    redirects = dict(redirects)
    for e, r in sorted(redirects.iteritems()):
        line = '%s\n' % '\t'.join([e] + r)
        fh.write(line.encode('utf8'))

if __name__ == '__main__':
    data = Data.read(sys.stdin)
    redirects = fetch(data)
    write(redirects)
