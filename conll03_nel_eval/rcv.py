from __future__ import print_function
import sys
import re
from collections import defaultdict
import zipfile
import argparse
import itertools
import os


class ReutersCodes(object):
    """Index CoNLL docs by Retuers topic, country and indexing codes

    The following will index Reuters docs by topic:
        %(prog)s path/to/rcv1 -f conll_ner/etc/files.eng.testb

    The following will prepare evaluation of the sport documents:
        KEEP_REGEX=$(%(prog)s path/to/rcv1 \\
                 -f path/to/conll_ner/etc/files.eng.testb \\
                 -m gold.testb.txt -r |
                 grep ^topics:GSPO | cut -f2)
         cne prepare -k "$KEEP_REGEX" gold.testb.txt > gold.testb.topics:GSPO
        cne prepare -k "$KEEP_REGEX" system.testb.txt > system.testb.topics:GSPO
    """
    def __init__(self, rcv1_dir, files=sys.stdin, map_via=None, as_regexp=False):
        self.rcv1_dir = rcv1_dir
        self.files = files
        self.map_via = map_via
        self.as_regexp = as_regexp

    def __call__(self):
        if self.map_via:
            mapping = (l[l.index('(')+1:l.rindex(')')]
                       for l in self.map_via if l.startswith('-DOCSTART-'))
        else:
            mapping = None

        regexp = re.compile('''(?:<codes class="bip:([^:]+?):|<code code="([^"]+?)")''')
        index = defaultdict(list)
        for date, paths in itertools.groupby(self.files, lambda l: l.split('/')[0]):
            with zipfile.ZipFile(os.path.join(self.rcv1_dir, date) + '.zip') as zf:
                for path in paths:
                    path = path.strip().split('/')[1]
                    f = zf.open(path)
                    doc_id = path if mapping is None else next(mapping)
                    text = f.read()
                    group = None
                    for new_group, code in regexp.findall(text):
                        if code:
                            index['{0}:{1}'.format(group, code)].append(doc_id)
                        else:
                            group = new_group

        res = []
        if self.as_regexp:
            fmt = '{}\t^({})$'
            sep = '|'
            fn = re.escape
        else:
            fmt = '{}\t{}'
            sep = '\t'
            fn = str
        for k, docs in sorted(index.iteritems()):
            res.append(fmt.format(k, sep.join(map(fn, docs))))
        return '\n'.join(res)

    @classmethod
    def add_arguments(cls, ap):
        ap.add_argument('rcv1_dir', help='RCV1 data directory containing data zipped by day')
        ap.add_argument('-f', '--files', type=argparse.FileType('r'), default=sys.stdin, help='CoNLL NER Reuters file mapping')
        ap.add_argument('-m', '--map-via', type=argparse.FileType('r'), default=None, help='Map RCV paths back to CoNLL-YAGO Doc IDs by providing an annotation file')
        ap.add_argument('-r', '--as-regexp', action='store_true', default=False, help='List doc IDs as a matching regexp, useful with -m for cne prepare -k')
        ap.set_defaults(cls=cls)
        return ap

