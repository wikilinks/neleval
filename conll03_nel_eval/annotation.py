#!/usr/bin/env python
from collections import OrderedDict

class Reader(object):
    def __init__(self, fh):
        self.fh = fh

    def __iter__(self):
        return self.read()

    def read(self):
        for docid, annots in self.group_by_docid(self.annotations()):
            yield Document(docid, sorted(annots))

    def group_by_docid(self, annotations):
        d = OrderedDict()
        for a in annotations:
            if a.docid in d:
                d[a.docid].append(a)
            else:
                d[a.docid] = [a]
        return d.iteritems()

    def annotations(self):
        "Yield Annotation objects"
        for line in self.fh:
            yield Annotation.from_string(line.rstrip('\n'))

class Annotation(object):
    def __init__(self, docid, start, end, candidates=[]):
        self.docid = docid
        self.start = start
        self.end = end
        self.candidates = candidates

    def __str__(self):
        return u'{}\t{}\t{}\t{}'.format(
            self.docid,
            self.start,
            self.end,
            u'\t'.join([str(c) for c in self.candidates])
            )

    def __cmp__(self, other):
        assert isinstance(other, Annotation)
        return cmp((self.start, -self.end), (other.start, -other.end))

    # Getters
    @property
    def link(self):
        if len(self.candidates) > 0:
            return self.candidates[0]

    @property
    def kbid(self):
        if self.link:
            return self.link.kbid

    @property
    def score(self):
        if self.link:
            return self.link.score

    @property
    def type(self):
        if self.link:
            return self.link.type

    @property
    def is_nil(self):
        if self.link is None:
            return True
        if self.kbid.startswith('NIL'):
            return True
        return False

    @property
    def is_linked(self):
        return not self.is_nil
        

    # Parsing methods
    @classmethod
    def from_string(cls, s):
        docid, start, end, candidates = None, None, None, []
        cols = s.rstrip('\n\t').split('\t', 3)
        if len(cols) < 3:
            raise SyntaxError('Annotation must have at least 3 columns')
        if len(cols) >= 3:
            docid = cols[0]
            start = int(cols[1])
            end = int(cols[2])
        if len(cols) == 4:
            candidates = sorted(Candidate.from_string(cols[3]), reverse=True)
        return Annotation(docid, start, end, candidates)

class Candidate(object):
    def __init__(self, kbid, score=None, type=None):
        self.kbid = kbid
        self.score = score
        self.type = type

    def __str__(self):
        return u'{}\t{}\t{}'.format(self.kbid,
                                    self.score or '',
                                    self.type or '')

    def __cmp__(self, other):
        assert isinstance(other, Candidate)
        return cmp(self.score, other.score)

    # Parsing methods
    @classmethod
    def from_string(cls, s):
        cols = s.rstrip('\t').split('\t')
        if len(cols) == 1:
            # link includes kbid only
            yield Candidate(cols[0])
        elif len(cols) == 2:
            # link includes kbid and score
            yield Candidate(cols[0], float(cols[1]))
        elif len(cols[3:]) % 3 == 0:
            # >=1 (kbid, score, type) candidate tuples
            for i in xrange(0, len(cols), 3):
                kbid, score, type = cols[i:i+3]
                yield cls(kbid, float(score), type)
        else:
            # undefined format
            raise SyntaxError('Need kbid, score and type when >1 candidates')

MATCHES = '''
strong_mention_match
strong_linked_mention_match
strong_link_match
entity_match
'''.strip().split()

TEMPLATE = u'{}\t{}\t{}\t{}\t{}'
ENC = 'utf8'

# Helper functions: key() and match()
def strong_key(i):
    return (i.start, i.end)

def strong_link_key(i):
    return (i.start, i.end, i.kbid)

def entity_key(i):
    return i

def weak_key(i):
    return list(xrange(i.start, i.end))

def weak_link_key(i):
    return [(j, i.kbid) for j in xrange(i.start, i.end)]

def weak_match(i, items, key_func):
    matches = []
    for i in key_func(i):
        res = items.get(i)
        if res is not None:
            matches.append(i)
    return matches

class Document(object):
    def __init__(self, id, annotations):
        self.id = id
        self.annotations = annotations

    def __str__(self):
        return u'\n'.join(str(a) for a in self.annotations)

    def __cmp__(self, other):
        return cmp(self.id, other.id)

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
        fn = [(i, None) for i in sorted(index.values())]
        return tp, fp, fn
