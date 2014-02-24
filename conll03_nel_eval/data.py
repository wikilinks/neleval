#!/usr/bin/env python
"""
Container for CoNLL03 NEL annotation
"""
import re

MATCHES = '''
strong_mention_match
weak_mention_match
strong_link_match
weak_link_match
link_entity_match
strong_all_match
weak_all_match
'''.strip().split()

TEMPLATE = u'{}\t{}\t{}\t{}\t{}'
ENC = 'utf8'

# Different styles of format.
class CoNLLDialect(object):
    DOC_ID = re.compile('^-DOCSTART- \((?P<doc_id>.*)\)$')
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
            # [iob, name, wikipedia_id, wikipedia_url, numeric_id, freebase_id]
            if len(line_bits) == 6 or len(line_bits) == 5:
                iob = line_bits[0]
                name = line_bits[1]
                link = line_bits[3].split('/')[-1]
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

class Sentence(object):
    def __init__(self, spans):
        assert spans and all(isinstance(s, Span) for s in spans), 'Invalid Spans {}'.format(spans)
        self.spans = spans

    def __unicode__(self):
        return u'\n'.join(unicode(s) for s in self.spans)

    def iter_entities(self):
        return iter(set(l.link for l in self.iter_links()))

    def iter_mentions(self):
        return self._iter_mentions(link=True, nil=True)

    def iter_links(self):
        return self._iter_mentions(link=True, nil=False)

    def iter_nils(self):
        return self._iter_mentions(link=False, nil=True)

    def _iter_mentions(self, link=True, nil=True):
        assert not (not link and not nil), 'Must filter some mentions.'
        for s in self.spans:
            if isinstance(s, Mention):
                if not link and not s.link:
                    continue
                if not nil and s.link is None:
                    continue
                yield s

# Helper functions: key() and match()
def strong_key(i):
    return [(i.start, i.end)]

def weak_key(i):
    return list(xrange(i.start, i.end))

def strong_link_key(i):
    return [(i.start, i.end, i.link)]

def weak_link_key(i):
    return [(j, i.link) for j in xrange(i.start, i.end)]

def entity_key(i):
    return [i]

def strong_match(i, items, key_func):
    keys = key_func(i)
    assert len(keys) == 1
    res = items.get(keys[0])
    if res is None:
        return []
    else:
        return [keys[0]]

def weak_match(i, items, key_func):
    matches = []
    for i in key_func(i):
        res = items.get(i)
        if res is not None:
            matches.append(i)
    return matches

class Document(object):
    def __init__(self, doc_id, sentences, dialect):
        self.doc_id = doc_id
        self.sentences = sentences
        self.dialect = dialect

    def __str__(self):
        return '<Doc {} sentences>'.format(len(self.sentences))

    # Extracting matching mentions.
    def strong_mention_match(self, other):
        return self._match(other, strong_key, strong_match, 'iter_mentions')

    def strong_link_match(self, other):
        return self._match(other, strong_link_key, strong_match, 'iter_links')
    
    def strong_nil_match(self, other):
        return self._match(other, strong_link_key, strong_match, 'iter_nils')

    def strong_all_match(self, other):
        return self._match(other, strong_link_key, strong_match, 'iter_mentions')

    def weak_mention_match(self, other):
        return self._match(other, weak_key, weak_match, 'iter_mentions')

    def weak_link_match(self, other):
        return self._match(other, weak_link_key, weak_match, 'iter_links')
    
    def weak_nil_match(self, other):
        return self._match(other, weak_key, weak_match, 'iter_nils')
    
    def weak_all_match(self, other):
        return self._match(other, weak_link_key, weak_match, 'iter_mentions')

    def link_entity_match(self, other):
        return self._match(other, entity_key, strong_match, 'iter_entities')

    def _match(self, other, key_func, match_func, items_func_name):
        """ Assesses the match between this and the other document. 
        * other (Document)
        * key_func (a function that takes an item, returns a list of valid keys)
        * match_func (a function that take an item, items to match against and the key_func)
        * items_func_name (the name of a function that is called on Sentences)

        Returns three lists of items:
        * tp [(item, other_item), ...]
        * fp [(None, other_item), ...]
        * fn [(item, None), ...]
        """
        assert isinstance(other, Document)
        assert len(self.sentences) == len(other.sentences), 'Must compare documents with same number of sentences.'
        tp, fp, fn = [], [], []
        for s, o_s in zip(self.sentences, other.sentences):
            # Build indices.
            items = {}
            for i in getattr(s, items_func_name)():
                for k in key_func(i):
                    items[k] = i
            # Check against other.
            for o_i in getattr(o_s, items_func_name)():
                matching_keys = match_func(o_i, items, key_func)
                if matching_keys:
                    # Assume that all keys match to the same mention (to handle strong and weak)
                    matching = {items.pop(k) for k in matching_keys}
                    assert len(matching) == 1
                    tp.append((list(matching)[0], o_i))
                else:
                    fp.append((None, o_i))
            fn.extend([(i, None) for i in set(items.values())])
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
                # Grab the sentence tokens.
                else:
                    j = 0 # The token index in the sentence.
                    sentence = []
                    m = None
                    while l:
                        token, iob, name, link, score = self.dialect.extract_link(l)
                        if iob is None or iob == 'O':
                            if m is not None:
                                sentence.append(m)
                                m = None
                            sentence.append(Token(j, j+1, token))
                        elif iob == 'B':
                            if m is not None:
                                sentence.append(m)
                                m = None
                            m = Mention(j, j+1, name, [token], link=link, score=score)
                        elif iob == 'I':
                            assert m is not None
                            m.texts.append(token)
                            m.end += 1
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
        return Document(doc_id, sentences, dialect=self.dialect)

class Writer(Dialected):
    def write(self, doc):
        print >> self.f, u'{}\n{}\n'.format(self.dialect.format_doc_id(doc.doc_id), 
                u'\n\n'.join(unicode(s) for s in doc.sentences)).encode(ENC)
