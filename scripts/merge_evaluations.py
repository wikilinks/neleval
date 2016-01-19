#!/usr/bin/env python
"""Merge multiple evaluation files into one with prefixed measure names

If directories are given, and --out-dir, will group by filename.

Example usage:
    ./scripts/merge_evaluations.py --label-re='[^/]+/?$' -x eval_merged -l =TEDL2015_neleval-no1331 --out-dir /tmp/foobar tac15data/TEDL2015_neleval-no1331 $(find tac15data/TEDL2015_neleval-no1331/00filtered/ -type d )
"""
from __future__ import print_function
import argparse
import os
import glob
import collections
import sys
import re

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('-o', '--out-dir', default=None)
ap.add_argument('-x', '--out-extension', default=None)
ap.add_argument('-l', '--label', dest='labels', action='append',
                type=lambda s: s.split('=', 1))
ap.add_argument('-r', '--label-re', default=None, type=re.compile)
ap.add_argument('--fmt', default='{label}/{{}}')
ap.add_argument('paths', nargs='+')
args = ap.parse_args()


def _swap_ext(name, new_ext):
    if new_ext is None:
        return name
    name, ext = os.path.splitext(name)
    return name + '.' + new_ext


nonexist = [path for path in args.paths if not os.path.exists(path)]
if nonexist:
    ap.error('Paths do not exist: %r' % nonexist)

is_dir = [os.path.isdir(path) for path in args.paths]
if all(is_dir):
    if args.out_dir is None:
        ap.error('Must specify --out-dir in path mode')
    input_paths = collections.defaultdict(list)
    for dir_path in args.paths:
        for path in glob.glob(os.path.join(dir_path, '*.evaluation')):
            input_paths[os.path.basename(path)].append(path)
    outputs = {name: os.path.join(args.out_dir,
                                  _swap_ext(name, args.out_extension))
               for name in input_paths}
elif not any(is_dir):
    if args.out_dir is not None or args.out_extension is not None:
        ap.error('--out-dir and --out-extension not used in files mode; output is STDOUT')
    input_paths = {'all': args.paths}
    outputs = {'all': sys.stdout}
else:
    ap.error('Got mixture of directories (e.g. %r) and files (e.g. %r)' % (args.paths[is_dir.index(True)], args.paths[is_dir.index(False)]))

seen_labels = set()
labels = {src: dst for dst, src in args.labels or []}

def get_label(path):
    name = os.path.dirname(path)
    if args.label_re:
        match = args.label_re.search(name)
        if match is not None:
            name = match.group()
    seen_labels.add(name)
    return labels.get(name, name)


for name in input_paths:
    fout = outputs[name]
    if not hasattr(fout, 'read'):
        opened = True
        fout = open(fout, 'w')
    else:
        opened = False
    print('Processing', name, 'to', fout.name, file=sys.stderr)
    for i, path in enumerate(input_paths[name]):
        label = get_label(path)
        if label:
            fmt = args.fmt.format(label=label)
        else:
            fmt = '{}'
        fmt = '{{}}\t{}'.format(fmt)
        with open(path) as fin:
            fin = iter(fin)
            try:
                header = next(fin)
            except StopIteration:
                print('Found empty file at', path, file=sys.stderr)
            if i == 0:
                fout.write(header)
            for l in fin:
                l, measure = l.rstrip('\n\r').rsplit('\t', 1)
                print(fmt.format(l, measure), file=fout)
    if opened:
        fout.close()

unseen_labels = set(labels) - seen_labels
if unseen_labels:
    print('WARNING: did not see labels %r' % sorted(unseen_labels), file=sys.stderr)
