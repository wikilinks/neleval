#!/usr/bin/env python
"Document - group and compare annotations"
from .annotation import Annotation
from collections import OrderedDict

TEMPLATE = u'{}\t{}\t{}\t{}\t{}'
ENC = 'utf8'


# Document class contains methods for linking annotation

class Document(object):
    def __init__(self, id, annotations):
        self.id = id
        self.annotations = annotations

    def __str__(self):
        return unicode(self)

    def __unicode__(self):
        return u'\n'.join(unicode(a) for a in self.annotations)

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
    return d.iteritems()

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
        for groupid, annots in self.group(self.annotations()):
            yield self.cls(groupid, annots)

    def annotations(self):
        "Yield Annotation objects"
        for line in self.fh:
            yield Annotation.from_string(line.rstrip('\n').decode(ENC))
