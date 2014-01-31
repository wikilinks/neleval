#!/usr/bin/env python
"""
Fetch redirects for entities in gold standard.

Call as, e.g.:
  cat /data/nel/conll03/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv \
    | python fetch_map.py \
    --split testa  \
    > maps/map-testa-fromapi-20140130.tsv
"""
import sys
import operator
from data import Data
from wikipedia import Wikipedia
from utils import log

def fetch(data, split=None):
    w = Wikipedia()
    seen = set()
    for d in data.documents.values():
        if split is not None and d.split != split:
            continue # doc not in specified split
        for e in d.entities:
            if e in seen:
                continue
            seen.add(e)
            current = w.redirected(e.decode('utf8')) # fetch current title
            redirects = w.redirects(current) # fetch all incoming redirects
            yield current, sorted(list(redirects))

def write(redirects, fh=sys.stdout):
    redirects = dict(redirects)
    for e, r in sorted(redirects.iteritems()):
        line = '%s\n' % '\t'.join([e] + r)
        fh.write(line.encode('utf8'))

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', help='Data split to process')
    args = ap.parse_args()
    data = Data.read(sys.stdin)
    redirects = fetch(data, split=args.split)
    write(redirects)
