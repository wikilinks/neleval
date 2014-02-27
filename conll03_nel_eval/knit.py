import re
import xml.etree.cElementTree as ET
from data import Reader, Document, Sentence, Mention, Token, Writer
from cStringIO import StringIO

class Tagme(object):
    def __init__(self, fname, tmxml, keep=None, threshold=0.0):
        self.fname = fname # aida/conll gold file
        self.tmxml = tmxml # tagme xml annotations file
        self.keep = re.compile(keep) if keep else None # e.g., .*testb.*
        self.thresh = threshold # e.g., 0.289 strong annot, 0.336 entity

    def __call__(self):
        self.docs = dict(self.docs_by_text())
        out = StringIO()
        w = Writer(out)
        for doc in self.knit():
            w.write(doc)
        return out.getvalue()

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('tagme',
                          help='Knit tagme annotations to aida/conll format')
        p.add_argument('fname', metavar='FILE')
        p.add_argument('-a', '--tmxml', help='tagme2 xml annotations file')
        p.add_argument('-k', '--keep', help='regex pattern to capture')
        p.add_argument('-t', '--threshold', type=float,
                       help='tagme2 confidence threshold')
        p.set_defaults(cls=cls)
        return p

    # METHODS TO READ AND STORE GOLD DOCUMENTS..

    def docs_by_text(self):
        """Yield (text, doc) tuples."""
        for doc in Reader(open(self.fname)):
            if self.keep and not self.keep.match(doc.doc_id):
                continue
            texts = self.doctexts(doc)
            text = ''.join(texts)
            yield text, (doc.doc_id, texts)

    def doctexts(self, doc):
        """Return token texts for given document."""
        # TODO move to data.Document
        texts = []
        for sent in doc.sentences:
            texts.extend(self.senttexts(sent))
        return texts

    def senttexts(self, sent):
        """Return token texts for given sentence."""
        # TODO move to data.Sentence
        texts = []
        for span in sent:
            texts.extend(span.text.split())
        return texts

    # METHODS TO READ AND KNIT TAGME OUTPUT TO CONLL TOKENISATION..

    def knit(self):
        """Yield document objects with CoNLL tokenisation and TagMe mentions."""
        for tagme_text, annots in self.read():
            assert tagme_text in self.docs, 'Could not find AIDA/CoNLL doc.'
            doc_id, doc_texts = self.docs[tagme_text]
            annots = dict(self.annots_by_tokid(annots, doc_texts))
            sent = self.sentence(annots, doc_texts)
            yield Document(doc_id, [sent])

    def read(self):
        """Yield (text, mention_list) tuples from TagMe output file."""
        tree = ET.parse(self.tmxml)
        root = tree.getroot()
        for instance in root.iter('instance'):
            yield self.parse(instance)

    def parse(self, instance):
        """Return text, mention_list for given doc element."""
        doc_texts = []
        mentions = []
        char_count = 0
        # parse any text before first annotation
        texts, char_count = self.tokens(instance.text, char_count)
        doc_texts.extend(texts)
        # parse annotations
        for annotation in instance.iter('annotation'):
            # annotation text
            m, char_count = self.mention(annotation, char_count)
            if m.score and self.thresh and m.score > self.thresh:
                mentions.append(m)
            doc_texts.extend(m.texts)
            # annotation tail
            texts, char_count = self.tokens(annotation.tail, char_count)
            doc_texts.extend(texts)
        text = ''.join(doc_texts)
        return text, mentions

    def tokens(self, text, char_count):
        toks = []
        if text is not None:
            for tok in unicode(text).split():
                char_count += len(tok)
                toks.append(tok)
        return toks, char_count

    def mention(self, elem, char_count):
        start = char_count
        name = unicode(elem.text).replace('\n', '<s/>')
        texts, char_count = self.tokens(elem.text, char_count)
        link = unicode(elem.get('title'))
        score = float(elem.get('score'))
        m = Mention(start, char_count, name, texts, link, score)
        return m, char_count

    def annots_by_tokid(self, annots, doc_texts):
        """Yield (startid, mention) tuples based on gold tokens."""
        tokids = list(self.tokid_per_char(doc_texts))
        for m in annots:
            m.start = tokids[m.start]
            m.end = tokids[m.end-1] + 1
            m.texts = doc_texts[m.start:m.end]
            yield m.start, m

    def tokid_per_char(self, texts):
        """Yield tokids once per character."""
        char_count = 0
        for i, t in enumerate(texts):
            start = char_count
            char_count += len(t)
            for j in xrange(start, char_count):
                yield i

    def sentence(self, annots, doc_texts):
        """Return sentence object over given mentions and tokens."""
        tok_count = 0
        sentence = []
        while tok_count < len(doc_texts):
            if tok_count in annots:
                m = annots[tok_count]
                sentence.append(m)
                tok_count += len(m.texts)
            else:
                text = doc_texts[tok_count]
                t = Token(tok_count, tok_count+1, text)
                sentence.append(t)
                tok_count += 1
        return Sentence(sentence)
