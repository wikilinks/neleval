from io import BytesIO
import re
import xml.etree.cElementTree as ET

from .data import Reader, Writer, ENC, Document, Sentence, Mention, Token


class Unstitch(object):
    'Produce release file from system output'

    def __init__(self, fname):
        """
        fname - system output
        """
        self.fname = fname

    def __call__(self):
        lines = []
        for doc in list(sorted(Reader(open(self.fname)))):
            for m in doc.iter_mentions():
                lines.append(u'{}\t{}\t{}\t{}\t{}'.format(doc.doc_id, m.start, m.end, m.link or '', m.score or ''))
        return '\n'.join(lines).encode(ENC)
            
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('fname', metavar='FILE')
        p.set_defaults(cls=cls)
        return p


class Stitch(object):
    'Merge release file with gold-standard'

    def __init__(self, fname, gold=None):
        """
        fname - system output (release format)
        gold - gold standard
        """
        self.fname = fname
        self.gold = gold

    def __call__(self):
        # Read release file.
        data = {}
        with open(self.fname) as f:
            for l in f:
                parts = l.decode(ENC).rstrip('\n').split('\t')
                doc_id = start = end = link = score = None
                if len(parts) == 4:
                    doc_id, start, end, link = parts
                elif len(parts) == 5:
                    doc_id, start, end, link, score = parts
                else:
                    raise ValueError('Expected 4 or 5 parts to the line, got {}'.format(parts))
                if not doc_id in data:
                    data[doc_id] = []
                data[doc_id].append((int(start), int(end), link or None, 
                                    float(score) if score else None))
        # Merge into docs. 
        docs = list(sorted(Reader(open(self.gold))))
        out = BytesIO()
        w = Writer(out)
        for doc in docs:
            doc.clear_mentions()
            doc.set_mentions(data.get(doc.doc_id, []))
            w.write(doc)
        return out.getvalue()
            
    @classmethod
    def add_arguments(cls, p):
        p.add_argument('fname', metavar='FILE')
        p.add_argument('-g', '--gold')
        p.set_defaults(cls=cls)
        return p


class Tagme(object):
    'Reformat tagme annotations to aida/conll format'

    def __init__(self, fname, tmxml, keep=None, threshold=0.0):
        self.fname = fname # aida/conll gold file
        self.tmxml = tmxml # tagme xml annotations file
        self.keep = re.compile(keep) if keep else None # e.g., .*testb.*
        self.thresh = threshold # e.g., 0.289 strong annot, 0.336 entity

    def __call__(self):
        self.docs = dict(self.docs_by_text())
        out = BytesIO()
        w = Writer(out)
        for doc in self.knit():
            w.write(doc)
        return out.getvalue()

    @classmethod
    def add_arguments(cls, p):
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
            if m.link and m.score and self.thresh and m.score > self.thresh:
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
        link = self.link(elem)
        score = float(elem.get('score'))
        m = Mention(start, char_count, name, texts, link, score)
        return m, char_count

    def link(self, elem):
        link = elem.get('title')
        if link:
            return '_'.join(unicode(elem.get('title')).split())

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
