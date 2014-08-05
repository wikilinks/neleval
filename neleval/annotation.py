#!/usr/bin/env python
"Representation of link standoff annotation"


class Annotation(object):
    __slots__ = ['docid', 'start', 'end', 'candidates']

    def __init__(self, docid, start, end, candidates=[]):
        self.docid = docid
        self.start = start
        self.end = end
        self.candidates = candidates

    def __str__(self):
        return unicode(self)

    def __unicode__(self):
        return u'{}\t{}\t{}\t{}'.format(
            self.docid,
            self.start,
            self.end,
            u'\t'.join([unicode(c) for c in self.candidates])
            )

    def __cmp__(self, other):
        assert isinstance(other, Annotation)
        return cmp((self.start, -self.end), (other.start, -other.end))

    # Getters
    @property
    def span(self):
        return (self.docid, self.start, self.end)

    @property
    def link(self):
        "Return top candidate"
        if self.candidates:
            return self.candidates[0]

    @property
    def eid(self):
        "Return link KB ID or NIL cluster ID (default cluster ID is None)"
        if self.link:
            return self.link.id

    @property
    def kbid(self):
        "Return link KB ID or None"
        if self.is_linked:
            return self.link.id

    @property
    def score(self):
        "Return link score or None"
        if self.is_linked:
            return self.link.score

    @property
    def type(self):
        "Return link score or None"
        if self.is_linked:
            return self.link.type

    @property
    def is_nil(self):
        if self.eid is None:
            return True
        if self.eid.startswith('NIL'):
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
    __slots__ = ['id', 'score', 'type']
    def __init__(self, id, score=None, type=None):
        self.id = id
        self.score = score
        self.type = type

    def __str__(self):
        return unicode(self)

    def __unicode__(self):
        return u'{}\t{}\t{}'.format(self.id,
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
            # link includes id only
            yield cls(cols[0])
        elif len(cols) == 2:
            # link includes id and score
            yield cls(cols[0], float(cols[1]))
        elif len(cols[3:]) % 3 == 0:
            # >=1 (id, score, type) candidate tuples
            for i in xrange(0, len(cols), 3):
                id, score, type = cols[i:i+3]
                yield cls(id, float(score), type)
        else:
            # undefined format
            raise SyntaxError('Need id, score and type when >1 candidates')
