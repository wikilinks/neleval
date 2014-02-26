#!/usr/bin/env python
"""
Container for CoNLL03 NEL annotation
"""
import re
from .utils import log

MATCHES = '''
strong_mention_match
strong_link_match
entity_match
'''.strip().split()

TEMPLATE = u'{}\t{}\t{}\t{}\t{}'
ENC = 'utf8'

# Different styles of format.
class CoNLLDialect(object):
    DOC_ID = re.compile('^-DOCSTART- \(?(?P<doc_id>[^\)]*)\)?$')
    NIL = '--NME--'
    def __init__(self):
        pass

    def extract_doc_id(self, line):
        m = self.DOC_ID.match(line)
        if m is not None:
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
            assert len(line_bits) >= 2
            # [iob, name, yago_id, wikipedia_url, wikipedia_id, freebase_id]
            if len(line_bits) == 6 or len(line_bits) == 5:
                iob = line_bits[0]
                name = line_bits[1]
                link = line_bits[3].split('/')[-1]
            # [iob, name, entity_id, score]
            elif len(line_bits) == 4:
                # TODO handle multiple entities with scores
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

    def __cmp__(self, other):
        return cmp(self.start, other.start) or cmp(self.end, other.end)

class Token(Span):
    __slots__ = ['text', 'start', 'end']
    def __init__(self, start, end, text):
        assert isinstance(text, unicode)
        super(Token, self).__init__(start, end)
        self.text = text

    def __str__(self):
        return TEMPLATE.format(self.text, '', '', '', '').encode(ENC)

class Mention(Span):
    """ A mention composed of 1+ tokens, possibly links. """
    __slots__ = ['texts', 'link', 'score', 'name']
    def __init__(self, start, end, name, texts, link=None, score=None):
        assert isinstance(texts, list) and all(isinstance(i, unicode) for i in texts), 'Invalid texts {}'.format(texts)
        assert name and isinstance(name, unicode)
        assert link is None or isinstance(link, unicode)
        super(Mention, self).__init__(start, end)
        self.texts = texts
        self.link = link
        self.name = name
        self.score = score

    def __unicode__(self):
        lines = []
        link = self.link or ''
        score = self.score or ''
        for i, t in enumerate(self.texts):
            if i == 0:
                iob = 'B'
            else:
                iob = 'I'
            lines.append(TEMPLATE.format(t, iob, self.name, link, score))
        return u'\n'.join(lines)

    @property
    def text(self):
        return u' '.join(self.texts)

class Sentence(object):
    def __init__(self, spans):
        assert spans and all(isinstance(s, Span) for s in spans), 'Invalid Spans {}'.format(spans)
        self.spans = spans

    def __unicode__(self):
        return u'\n'.join(unicode(s) for s in self.spans)

    def __iter__(self):
        return iter(self.spans)

# Helper functions: key() and match()
def strong_key(i):
    return (i.start, i.end)

def strong_link_key(i):
    return (i.start, i.end, i.link)

def entity_key(i):
    return i

def weak_key(i):
    return list(xrange(i.start, i.end))

def weak_link_key(i):
    return [(j, i.link) for j in xrange(i.start, i.end)]

def weak_match(i, items, key_func):
    matches = []
    for i in key_func(i):
        res = items.get(i)
        if res is not None:
            matches.append(i)
    return matches

class Document(object):
    def __init__(self, doc_id, sentences):
        self.doc_id = doc_id
        self.sentences = sentences

    def __str__(self):
        return '<Doc {} sentences>'.format(len(self.sentences))

    def __cmp__(self, other):
        return cmp(self.doc_id, other.doc_id)

    # Accessing Spans.
    def _iter_mentions(self, link=True, nil=True):
        assert not (not link and not nil), 'Must filter some mentions.'
        for sentence in self.sentences:
            for s in sentence:
                if isinstance(s, Mention):
                    if not link and not s.link:
                        continue
                    if not nil and s.link is None:
                        continue
                    yield s

    def iter_mentions(self):
        return self._iter_mentions(link=True, nil=True)

    def iter_links(self):
        return self._iter_mentions(link=True, nil=False)

    def iter_nils(self):
        return self._iter_mentions(link=False, nil=True)

    def iter_entities(self):
        return iter(set(l.link for l in self.iter_links()))

    # Matching mentions.
    def strong_mention_match(self, other):
        return self._match(other, strong_key, 'iter_mentions')

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
        assert self.doc_id == other.doc_id, 'Must compare same document: {} vs {}'.format(self.doc_id, other.doc_id)
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

## Readers and writers.
class Dialected(object):
    def __init__(self, f, dialect_name=DEFAULT_DIALECT):
        assert dialect_name in DIALECTS, 'Invalid dialect "{}"'.format(dialect_name)
        self.dialect = DIALECTS[dialect_name]()
        self.f = f

class Reader(Dialected):
    def __iter__(self):
        return self.read()

    def read(self):
        ''' Yields Documents. '''
        doc_id = None
        sentences = []
        sentence = []
        m = None
        # Assume we can read all data into memory.
        lines = ((i, l) for i, l in enumerate(l.strip().decode('utf8') for l in self.f))
        while True:
            try:
                i, l = lines.next()
                temp_doc_id = self.dialect.extract_doc_id(l)
                if temp_doc_id:
                    if doc_id and sentences:
                        yield self.build_doc(doc_id, sentences)
                    doc_id = temp_doc_id
                    sentences = []
                    j = 0 # The token index in the document.
                # Grab the sentence tokens.
                else:
                    sentence = []
                    m = None
                    while l:
                        try:
                            token, iob, name, link, score = self.dialect.extract_link(l)
                        except AssertionError, e:
                            log('Error reading line {}\t{}'.format(i, e))
                            raise e
                        if iob is None or iob == 'O':
                            if m is not None:
                                sentence.append(m)
                                m = None
                            sentence.append(Token(j, j+1, token))
                        elif m is not None and iob == 'I':
                            m.texts.append(token)
                            m.end += 1
                        elif iob in 'IB':
                            if m is not None:
                                sentence.append(m)
                                m = None
                            m = Mention(j, j+1, name, [token], link=link, score=score)
                        else:
                            assert False, 'Unexpected IOB case "{}"'.format(iob)
                        i, l = lines.next()
                        j += 1
                    if m is not None:
                        sentence.append(m)
                        m = None
                    if sentence:
                        sentences.append(Sentence(sentence))
                        sentence = []
            except StopIteration:
                # Handle last document.
                if m is not None:
                    sentence.append(m)
                if sentence:
                    sentences.append(Sentence(sentence))
                yield self.build_doc(doc_id, sentences)
                break

    def build_doc(self, doc_id, sentences):
        return Document(doc_id, sentences)

class Writer(Dialected):
    def write(self, doc):
        print >> self.f, u'{}\n{}\n'.format(self.dialect.format_doc_id(doc.doc_id), 
                u'\n\n'.join(unicode(s) for s in doc.sentences)).encode(ENC)
