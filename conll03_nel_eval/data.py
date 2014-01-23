#!/usr/bin/env python
"""
Container for CoNLL03 NEL annotation
"""

class Mention(object):
    """Named entity mention."""
    def __init__(self, start, end, link=None):
        """
        start - begin token offset (slice semantics)
        end - end token offset (slice semantics)
        link - e.g., Wikipedia title (None == NIL)
        """
        self.start = start
        self.end = end
        self.link = link

    def __str__(self):
        return '<Mention [{}:{}] -> {}>'.format(self.start, self.end, self.link)

    def __cmp__(self, other):
        return cmp(self.start, other.start) or cmp(self.end, other.end)

    def span_match(self, other):
        return cmp(self, other) == 0

    def fuzzy_span_match(self, other):
        raise NotImplementedError

    def link_match(self, other):
        return self.link == other.link

class Document(object):
    def __init__(self, id, mentions=None):
        self.id, self.split = self._parse_id(id)
        self.mentions = mentions or []

    def _parse_id(self, id):
        split = 'train'
        if 'testa' in id:
            split = 'testa'
        elif 'testb' in id:
            split = 'testb'
        id = id.replace(split, '')
        return id, split

    @classmethod
    def from_string(cls, s):
        raise NotImplementedError
    
class Data(object):

    @classmethod
    def from_string(cls, s):
        raise NotImplementedError
