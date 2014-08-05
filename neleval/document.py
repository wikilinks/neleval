#!/usr/bin/env python
"Document - group and compare annotations"
from .annotation import Annotation
from collections import OrderedDict, Sequence
import operator

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

    def count_matches(self, resp, match):
        if not hasattr(match, 'build_index'):
            match = LMATCH_DEFS[match]
        gold_index = match.build_index(self.annotations)
        resp_index = match.build_index(resp.annotations)
        tp = len(keys(gold_index) & keys(resp_index))
        fn = len(gold_index) - tp
        fp = len(resp_index) - tp
        return tp, fp, fn

    def get_matches(self, resp, match):
        """ Assesses the match between this and the other document.
        * resp (response document for self being gold)
        * match (MatchDef object or name)

        Returns three lists of items:
        * tp [(item, other_item), ...]
        * fp [(None, other_item), ...]
        * fn [(item, None), ...]
        """
        if not hasattr(match, 'build_index'):
            match = LMATCH_DEFS[match]
        gold_index = match.build_index(self.annotations)
        resp_index = match.build_index(resp.annotations)
        gold_keys = keys(gold_index)
        resp_keys = keys(resp_index)
        shared = gold_keys & resp_keys
        tp = [(gold_index[k], resp_index[k]) for k in shared]
        fp = [(None, resp_index[k]) for k in resp_keys - shared]
        fn = [(gold_index[k], None) for k in gold_keys - shared]
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
        key = a.span # TODO should be strong_typed_key for tac?
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
