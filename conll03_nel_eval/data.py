#!/usr/bin/env python
"""
Container for CoNLL03 NEL annotation
"""
import gzip
import re
from collections import OrderedDict

MATCHES = '''
strong_mention_match
weak_mention_match
strong_link_match
weak_link_match
strong_nil_match
weak_nil_match
link_entity_match
strong_all_match
weak_all_match
'''.strip().split()

TEMPLATE = u'{}\t{}\t{}\t{}'
ENC = 'utf8'

# Different styles of format.
class CoNLLDialect(object):
    DOC_ID = re.compile('^-DOCSTART- \((?<doc_id>.*)\)$')
    NIL = '--NME--'
    def __init__(self):
        pass

    def extract_doc_id(self, line):
        m = self.DOC_ID.match(line)
        if m is not None:
            # TODO Splits?
            return m.group(1)
    
    def format_doc_id(self, doc_id):
        return '-DOCSTART- ({})'.format(doc_id)

    def extract_link(self, line):
        line_bits = line.split('\t')
        token = line_bits.pop(0)
        iob = None
        name = None
        link = None
        score = None
        if line_bits:
            assert len(line_bits) >= 2:
            # [iob, name, wikipedia_id, wikipedia_url, numeric_id, freebase_id]
            if len(line_bits) == 6:
                iob = line_bits[0]
                name = line_bits[1]
                # FIXME Configurable?
                link = line_bits[2]
            # [iob, name, entity_id, score]
            elif len(line_bits) == 4:
                iob = line_bits[0]
                name = line_bits[1]
                link = line_bits[2]
                score = float(line_bits[3])
            # [iob, name, entity_id]
            elif len(line_bits) == 3:
                iob = line_bits[0]
                name = line_bits[1]
                link = line_bits[2]
            # [iob, name] - this may be non-standard...
            elif len(line_bits) == 2:
                iob = line_bits[0]
                name = line_bits[1]
            else:
                assert False, 'Impossible choice {}'.format(line_bits)
        if link == self.NIL:
            link = None
        return token, iob, name, link, score

DIALECTS = {
    'CoNLL': CoNLLDialect,
}
DEFAULT_DIALECT = 'CoNLL'

# Data model.
class Span(object):
    __slots__ = ['start', 'end']
    def __init__(self, start, end):
        assert isinstance(start, int) and 0 <= start
        assert isinstance(end, int) and start < end
        self.start = start
        self.end = end

class Token(Span):
    __slots__ = ['text', 'start', 'end']
    def __init__(self, start, end, text):
        assert isinstance(text, unicode)
        super(Token, self).__init__(start, end)
        self.text = text

    def __str__(self):
        return TEMPLATE.format(self.text).encode(ENC)

class Mention(object):
    """ A mention composed of 1+ tokens, possibly links. """
    __slots__ = ['texts', 'link', 'score']
    def __init__(self, start, end, texts, link=None, score=None):
        assert isinstance(texts, list) and all(isinstance(unicode, i) for i in texts)
        assert link is None or isinstance(link, unicode)
        super(Mention, self).__init__(start, end)
        self.texts = texts
        self.link = link
        self.score = score

    def __str__(self):
        lines = []
        link = self.link or ''
        score = self.score or ''
        for i, t in enumerate(self.texts):
            if i == 0:
                iob = 'B'
            else:
                iob = 'I'
            lines.append(TEMPLATE.format(t, iob, link, score))
        return u'\n'.join(lines).encode(ENC)

class Sentence(object):
    def __init__(self, spans):
        assert spans and all(isinstance(Span, s) for s in spans)
        self.spans = spans

    def __str__(self):
        return u'\n'.join(unicode(s) for s in self.spans)

class Document(object):
    def __init__(self, doc_id, sentences, dialect):
        self.doc_id = doc_id
        self.sentences = sentences
        self.dialect = dialect

    def __str__(self):
        return u'{}{}\n'.format(self.doc_id, u'\n'.join(unicode(s) for s in self.sentences)).encode(ENC)

class Parser(object):
    def __init__(self, dialect_name=DEFAULT_DIALECT):
        assert dialect_name in DIALECTS, 'Invalid dialect "{}"'.format(dialect_name)
        self.dialect = DIALECTS[dialect_name]

    def parse(self, f):
        ''' Yields Documents. '''
        doc_id = None
        sentences = []
        lines = (enumerate(l.strip() for l in f))
        try:
            i, l = lines.next()
            temp_doc_id = self.dialect.extract_doc_id(l)
            if temp_doc_id:
                yield self.build_doc(doc_id, sentences)
                doc_id = temp_doc_id
                sentences = []
            # Grab the sentence tokens.
            else:
                j = 0 # The token index in the sentence.
                while l:
                    token, iob, name, link, score = self.dialect.extract_link(l)
                    # TODO Pack tokens into mentions.
                    i, l = lines.next()
                    j += 1
        except StopIteration:
            # Handle last document.
            yield self.build_doc(doc_id, sentences)
    
    def build_doc(self, doc_id, sentences):
        return Doc(doc_id, sentences, dialect=self.dialect)
        
# FIXME Let's keep matchers out of the data model.

class OldMention(object):
    """Named entity mention."""
    def __init__(self, start, end, link=None):
        """
        start - begin token offset (slice semantics)
        end - end token offset (slice semantics)
        link - e.g., Wikipedia url (None == NIL)
        """
        assert isinstance(start, int) and isinstance(end, int)
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
        return '<OldMention [{}:{}] -> {}>'.format(self.start, self.end, self.link)

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
        return self.title == other.title

    def strong_link_match(self, other):
        return self.strong_span_match(other) and self.link_match(other)

    def weak_link_match(self, other):
        return self.weak_span_match(other) and self.link_match(other)

class OldDocument(object):
    def __init__(self, id, mentions=None, lines=None):
        self.id, self.split = self._parse_id(id)
        self.mentions = mentions or []
        self.lines = lines or [] # FIXME Temporary inclusion of lines until tokens.

    def __str__(self):
        return '<OldDocument id={}>'.format(self.id)

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
                entities.add(m.title)
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

    def strong_all_match(self, other):
        return self._all_match(other, self.strong_link_match, self.strong_nil_match)

    def weak_all_match(self, other):
        return self._all_match(other, self.weak_link_match, self.weak_nil_match)

    def _all_match(self, other, l_func, n_func):
        l_tp, l_fp = l_func(other)
        n_tp, n_fp = n_func(other)
        return l_tp + n_tp, l_fp + n_fp

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

    def link_entity_match(self, other):
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
        start, link, prev = None, None, None
        for line in lines:
            if line.strip() == '' or line.startswith(cls.START):
                continue # blank line
            tok, bi, name, link = cls._parse_line(line)
            if start is not None and bi != 'I':
                yield OldMention(start, tok_id+1, prev)
                start = None
            tok_id += 1
            prev = link
            if bi is None:
                continue # not an entity mention
            if bi == 'B':
                start = tok_id
        if start is not None:
            yield OldMention(start, tok_id+1, prev)

    NIL_LINK = '--NME--'
    @classmethod
    def _parse_line(cls, line):
        cols = line.rstrip().split('\t')
        tok, bi, name, link = None, None, None, None
        if len(cols) >= 1:
            tok = cols[0] # current token
        if len(cols) >= 3:
            bi = cols[1] # begin/inside tag
            name = cols[2] # full mention text
        if len(cols) >= 5:
            link = cols[4] # Wikipedia url
        elif len(cols) == 4 and cols[3] != '--NME--':
            link = cols[3] # System output may not contain all items
        return tok, bi, name, link

    def to_conll(self, mapping=None):
        ''' Returns a CoNLL formatted string. '''
        if mapping:
            pass #TODO Actually tweak here, requires proper modelling of OldMentions.
        return ''.join(self.lines)


class Data(object):
    def __init__(self, documents=None):
        self.documents = documents or OrderedDict()

    @classmethod
    def read(cls, f):
        """
        Return data object.
        f - CoNLL-formatted file
        """
        if not hasattr(f, 'read'):
            if f.endswith('.gz'):
                f = gzip.open(f)
            else:
                f = open(f)
        data = cls()
        for d in cls._iter_documents(f):
            data.documents[d.id] = d
        return data

    @classmethod
    def _iter_documents(cls, lines):
        doc = []
        for i, line in enumerate(lines):
            if line.startswith('-DOCSTART-'):
                if len(doc) > 0:
                    yield OldDocument.from_lines(doc)
                doc = []
            if line.strip() != '':
                doc.append(line)
        yield OldDocument.from_lines(doc)

    def __iter__(self):
        return iter(self.documents.values())

    def __len__(self):
        return len(self.documents)
