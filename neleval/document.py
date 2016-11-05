#!/usr/bin/env python
"Document - group and compare annotations"

from __future__ import print_function

from collections import OrderedDict
import warnings
import sys
import argparse

from .annotation import Annotation
from .utils import unicode

TEMPLATE = u'{}\t{}\t{}\t{}\t{}'
ENC = 'utf8'


# Document class contains methods for linking annotation

class Document(object):
    def __init__(self, id, annotations):
        self.id = id
        self.annotations = sorted(annotations, key=lambda a: (a.start, -a.end))
        self._validate()
        self._set_fields()

    # May be 'ignore', 'warn', 'error'
    VALIDATION = {
        'nested': 'ignore',
        'crossing': 'ignore',
        'duplicate': 'ignore',
    }

    def _validate(self, _categories=['nested', 'crossing', 'duplicate']):
        # XXX: do we nee to ensure start > end for all Annotations first?
        issues = {cat: [] for cat, val in self.VALIDATION.items()
                  if val != 'ignore'}
        if not issues:
            return
        open_anns = []
        tags = sorted([(a.start, 'open', a) for a in self.annotations] +
                      [(a.end + 1, 'close', a) for a in self.annotations])  # use stop, not end
        for key, op, ann in tags:
            if op == 'open':
                open_anns.append(ann)
            else:
                open_anns.remove(ann)
                for other in open_anns:
                    comparison = ann.compare_spans(other)
                    if comparison in issues:
                        issues[comparison].append((other, ann))
        for issue, instances in issues.items():
            if not instances:
                continue
            if self.VALIDATION[issue] == 'error':
                b, a = instances[0]
                raise ValueError(u'Found annotations with {} span:\n{}\n{}'
                                 .format(issue, unicode(a), unicode(b)))
            elif self.VALIDATION[issue] == 'warn':
                b, a = instances[0]
                warnings.warn(u'Found annotations with {} span:\n{}\n{}'
                              .format(issue, unicode(a), unicode(b)))

    def _set_fields(self):
        """Set fields on annotations that are relative to document"""
        seen = set()
        for a in self.annotations:
            eid = a.eid
            a.is_first = eid not in seen
            seen.add(eid)

    def __unicode__(self):
        return u'\n'.join(unicode(a) for a in self.annotations)

    __str__ = __unicode__

    # Accessing Spans.
    def _iter_mentions(self, link=True, nil=True):
        assert not (not link and not nil), 'Must filter some mentions.'
        for a in self.annotations:
            #TODO check logic, handle TAC NILs
            if not link and a.is_linked:
                continue # filter linked mentions
            if not nil and a.is_nil:
                continue # filter nil mentions
            yield a

    def iter_mentions(self):
        return self._iter_mentions(link=True, nil=True)

    def iter_links(self):
        return self._iter_mentions(link=True, nil=False)

    def iter_nils(self):
        return self._iter_mentions(link=False, nil=True)


# Grouping annotations

def by_document(annotations):
    d = OrderedDict()
    for a in annotations:
        if a.docid in d:
            d[a.docid].append(a)
        else:
            d[a.docid] = [a]
    return d.items()


def by_mention(annotations):
    return [("{}//{}..{}".format(a.docid, a.start, a.end), [a])
            for a in annotations]


# Reading annotations

class Reader(object):
    "Read annotations, grouped into documents"
    def __init__(self, fh, group=by_document, cls=Document):
        self.fh = fh
        self.group = group
        self.cls = cls

    def __iter__(self):
        return self.read()

    def read(self):
        try:
            for groupid, annots in self.group(self.annotations()):
                yield self.cls(groupid, annots)
        except Exception:
            print('ERROR while processing', self.fh, file=sys.stderr)
            raise

    def annotations(self):
        "Yield Annotation objects"
        for line in self.fh:
            yield Annotation.from_string(line.rstrip('\n'))


class ValidateSpans(object):
    """Identify duplicate, crossing and nested spans

    Will output warnings or errors as determined by options.
    """
    def __init__(self, system, duplicate='error', crossing='warn', nested='ignore'):
        self.system = system
        self.duplicate = duplicate
        self.crossing = crossing
        self.nested = nested

    def __call__(self):
        # HACK (not concurrency-safe)
        OLD_VALIDATION = Document.VALIDATION
        Document.VALIDATION = {k: getattr(self, k)
                               for k in OLD_VALIDATION}
        list(Reader(self.system))
        Document.VALIDATION = OLD_VALIDATION

    @classmethod
    def add_arguments(cls, p):
        CHOICES = ['ignore', 'warn', 'error']
        p.add_argument('system', nargs='?', default=sys.stdin, type=argparse.FileType('r'), metavar='FILE')
        p.add_argument('--duplicate', default='error', choices=CHOICES)
        p.add_argument('--crossing', default='warn', choices=CHOICES)
        p.add_argument('--nested', default='ignore', choices=CHOICES)
        p.set_defaults(cls=cls)
        return p
