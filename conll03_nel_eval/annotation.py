#!/usr/bin/env python

class AnnotationReader(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        return self.read()

    def read(self):
        for line in open(self.fname):
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

    # Parsing methods
    @classmethod
    def from_string(cls, s):
        docid, start, end, candidates = None, None, None, []
        cols = s.split('\t', 3)
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

    def __cmp__(self, other):
        assert isinstance(other, Candidate)
        return cmp(self.score, other.score)

    def __str__(self):
        return u'{}\t{}\t{}'.format(self.kbid, self.score, self.type or '')

    # Parsing methods
    @classmethod
    def from_string(cls, s):
        cols = s.split('\t')
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
