#!/usr/bin/env python
"""
Container for CoNLL03 NEL annotation
"""
import gzip
from collections import OrderedDict

class Mention(object):
    """Named entity mention."""
    def __init__(self, start, end, link=None):
        """
        start - begin token offset (slice semantics)
        end - end token offset (slice semantics)
        link - e.g., Wikipedia url (None == NIL)
        """
        self.start = start
        self.end = end
        self.link = link

    @property
    def title(self):
        if self.link is None:
            return None
        else:
            return self.link[self.link.rfind('/')+1:].replace('_', ' ')

    def __str__(self):
        return '<Mention [{}:{}] -> {}>'.format(self.start, self.end, self.link)

    def __cmp__(self, other):
        return cmp(self.start, other.start) or cmp(self.end, other.end)

    def strong_span_match(self, other):
        return cmp(self, other) == 0

    def weak_span_match(self, other):
        for i in xrange(self.start, self.end):
            for j in xrange(other.start, other.end):
                if i == j:
                    return True
        return False

    def link_match(self, other):
        return self.link == other.link

    def strong_link_match(self, other):
        return self.strong_span_match(other) and self.link_match(other)

    def weak_link_match(self, other):
        return self.weak_span_match(other) and self.link_match(other)

class Document(object):
    def __init__(self, id, mentions=None, lines=None):
        self.id, self.split = self._parse_id(id)
        self.mentions = mentions or []
        self.lines = lines or [] # FIXME Temporary inclusion of lines until tokens.

    def __str__(self):
        return '<Document id={}>'.format(self.id)

    @property
    def links(self):
        links = []
        for m in self.mentions:
            if m.link is not None:
                links.append(m)
        return links

    @property
    def nils(self):
        nils = []
        for m in self.mentions:
            if m.link is None:
                nils.append(m)
        return nils

    @property
    def entities(self):
        entities = set()
        for m in self.mentions:
            if m.link is not None:
                entities.add(m.link)
        return entities

    def strong_mention_match(self, other):
        return self.mention_match(other, 'mentions', 'strong_span_match')

    def weak_mention_match(self, other):
        return self.mention_match(other, 'mentions', 'weak_span_match')

    def strong_link_match(self, other):
        return self.mention_match(other, 'links', 'strong_link_match')

    def weak_link_match(self, other):
        return self.mention_match(other, 'links', 'weak_link_match')

    def strong_nil_match(self, other):
        return self.mention_match(other, 'nils', 'strong_span_match')

    def weak_nil_match(self, other):
        return self.mention_match(other, 'nils', 'weak_span_match')

    def mention_match(self, other, mtype, match):
        tp, fp = 0, 0
        for m in getattr(self, mtype):
            matches = []
            for o in getattr(other, mtype):
                if getattr(m, match)(o):
                    matches.append(o)
            if len(matches) == 0:
                fp += 1
            else:
                tp += 1
        return tp, fp

    def entity_match(self, other):
        tp, fp = 0, 0
        for e in self.entities:
            if e in other.entities:
                tp += 1
            else:
                fp += 1
        return tp, fp

    def _parse_id(self, id):
        split = 'train'
        if 'testa' in id:
            split = 'testa'
        elif 'testb' in id:
            split = 'testb'
        return id, split

    @classmethod
    def from_lines(cls, lines):
        """
        Return document object.
        s - CoNLL-formatted lines
        """
        doc = cls._init_document(lines)
        if doc is not None:
            doc.mentions = list(cls._iter_mentions(lines))
        return doc

    START = '-DOCSTART-'
    @classmethod
    def _init_document(cls, lines):
        for line in lines:
            if line.startswith(cls.START):
                id = line.strip().replace(cls.START, '').strip().lstrip('(').rstrip(')')
                return cls(id, lines=lines)

    @classmethod
    def _iter_mentions(cls, lines):
        tok_id = -1
        start = None
        for line in lines:
            if line.strip() == '' or line.startswith(cls.START):
                continue # blank line
            tok_id += 1
            tok, bi, name, link = cls._parse_line(line)
            if bi is None:
                continue # not an entity mention
            if bi == 'B':
                start = tok_id
            if tok == name.split()[-1]:
                yield Mention(start, tok_id+1, link)
                start = None

    @classmethod
    def _parse_line(cls, line):
        cols = line.rstrip('\n').split('\t')
        tok, bi, name, link = None, None, None, None
        if len(cols) >= 1:
            tok = cols[0] # current token
        if len(cols) >= 3:
            bi = cols[1] # begin/inside tag
            name = cols[2] # full mention text
        if len(cols) >= 5:
            link = cols[4] # Wikipedia url
        return tok, bi, name, link

    def to_conll(self):
        ''' Returns a CoNLL formatted string. '''
        return ''.join(self.lines)


class Data(object):
    def __init__(self, documents=None):
        self.documents = documents or OrderedDict()

    @classmethod
    def from_file(cls, f):
        """
        Return data object.
        f - CoNLL-formatted file
        """
        fh = gzip.open(f) if f.endswith('.gz') else open(f)
        data = cls()
        for d in cls._iter_documents(fh.readlines()):
            data.documents[d.id] = d
        return data

    @classmethod
    def _iter_documents(cls, lines):
        doc = []
        for i, line in enumerate(lines):
            if line.startswith('-DOCSTART-'):
                if len(doc) > 0:
                    yield Document.from_lines(doc)
                doc = []
            if line.strip() != '':
                doc.append(line)
        yield Document.from_lines(doc)

    def __iter__(self):
        return iter(self.documents.values())

    def __len__(self):
        return len(self.documents)
