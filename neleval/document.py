#!/usr/bin/env python
"Document - group and compare annotations"
from .annotation import Annotation
from collections import OrderedDict

ALL_LMATCHES = 'all'
CORNOLTI_WWW13_LMATCHES = 'cornolti'
HACHEY_ACL14_LMATCHES = 'hachey'
TAC_LMATCHES = 'tac'
TAC14_LMATCHES = 'tac14'
LMATCH_SETS = {
    ALL_LMATCHES: [
        'strong_mention_match',
        'strong_linked_mention_match',
        'strong_link_match',
        'strong_nil_match',
        'strong_all_match',
        'strong_typed_all_match',
        'entity_match',
        ],
    CORNOLTI_WWW13_LMATCHES: [
        'strong_linked_mention_match',
        'strong_link_match',
        'entity_match',
        ],
    HACHEY_ACL14_LMATCHES: [
        'strong_mention_match', # full ner
        'strong_linked_mention_match',
        'strong_link_match',
        'entity_match',
        ],
    TAC_LMATCHES: [
        'strong_link_match', # recall equivalent to kb accuracy before 2014
        'strong_nil_match', # recall equivalent to nil accuracy before 2014
        'strong_all_match', # equivalent to overall accuracy before 2014
        'strong_typed_all_match',  # wikification f-score for TAC 2014
        ],
    TAC14_LMATCHES: [
        'strong_typed_all_match', # wikification f-score for TAC 2014
        ]
    }
DEFAULT_LMATCH_SET = HACHEY_ACL14_LMATCHES

TEMPLATE = u'{}\t{}\t{}\t{}\t{}'
ENC = 'utf8'


# Helper functions for indexing annotations

def strong_key(i):
    return (i.docid, i.start, i.end)

def strong_link_key(i):
    return (i.start, i.end, i.kbid)

def strong_typed_link_key(i):
    return (i.start, i.end, i.kbid, i.type)

def entity_key(i):
    return i

def weak_key(i):
    return list(xrange(i.start, i.end))

def weak_link_key(i):
    return [(j, i.kbid) for j in xrange(i.start, i.end)]


# Helper function for matching annotations

def weak_match(i, items, key_func):
    matches = []
    for i in key_func(i):
        res = items.get(i)
        if res is not None:
            matches.append(i)
    return matches


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

    def iter_entities(self):
        return iter(set(l.kbid for l in self.iter_links()))

    # Matching mentions.
    def strong_mention_match(self, other):
        return self._match(other, strong_key, 'iter_mentions')

    def strong_linked_mention_match(self, other):
        return self._match(other, strong_key, 'iter_links')

    def strong_link_match(self, other):
        return self._match(other, strong_link_key, 'iter_links')

    def strong_nil_match(self, other):
        return self._match(other, strong_key, 'iter_nils')

    def strong_all_match(self, other):
        return self._match(other, strong_link_key, 'iter_mentions')

    def strong_typed_all_match(self, other):
        return self._match(other, strong_typed_link_key, 'iter_mentions')

    def weak_mention_match(self, other):
        raise NotImplementedError('See #26')
        # TODO Weak match: calculate TP and FN based on gold mentions
        return self._match(other, weak_key, weak_match, 'iter_mentions')

    def weak_link_match(self, other):
        raise NotImplementedError('See #26')
        return self._match(other, weak_link_key, weak_match, 'iter_links')

    def entity_match(self, other):
        return self._match(other, entity_key, 'iter_entities')

    def _match(self, other, key_func, items_func_name):
        """ Assesses the match between this and the other document.
        * other (Document)
        * key_func (a function that takes an item, returns a key)
        * items_func (the name of a function that is called on Sentences)

        Returns three lists of items:
        * tp [(item, other_item), ...]
        * fp [(None, other_item), ...]
        * fn [(item, None), ...]
        """
        assert isinstance(other, Document)
        assert self.id == other.id, 'Must compare same document: {} vs {}'.format(self.id, other.id)
        tp, fp = [], []

        # Index document items.
        index = {key_func(i): i for i in getattr(self, items_func_name)()}

        for o_i in getattr(other, items_func_name)():
            k = key_func(o_i)
            i = index.pop(k, None)
            # Matching - true positive.
            if i is not None:
                tp.append((i, o_i))
            # Unmatched in other - false positive.
            else:
                fp.append((None, o_i))
        fn = [(i, None) for i in index.values()]
        return tp, fp, fn


# Grouping annotations

def by_document(annotations):
    d = OrderedDict()
    for a in annotations:
        if a.docid in d:
            d[a.docid].append(a)
        else:
            d[a.docid] = [a]
    return d.iteritems()

def by_entity(annotations):
    d = {}
    for a in annotations:
        key = strong_key(a) # TODO should be strong_typed_key for tac?
        if a.eid in d:
            d[a.eid].add(key)
        else:
            d[a.eid] = {key}
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
