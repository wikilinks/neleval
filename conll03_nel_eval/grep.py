from __future__ import print_function
import sys
import io
import re
from copy import copy
import textwrap
import argparse
import itertools

from .data import Reader, Writer, Document, Sentence, Mention


# TODO: something with char encodings


def _read_aux(f):
    # generates a list of lines per doc
    lines = []
    for l in f:
        if l.startswith('-DOCSTART-') or l.startswith('-X-'):
            if lines:
                yield lines
                lines = []
            continue
        l = l.split()
        if not l:
            continue
        lines.append(l)
    if lines:
        yield lines


class Grep(object):
    """Limit annotated mentions to those matching a criterion

    This either matches a regular expression against the mention text, or
    accepts an auxiliary file, such as the CoNLL 2003 NER-annotated training
    data. Only system mentions whose metadata matches the expression will be
    output, along with all tokens.

    For example, to retain only mentions containing china, use: `
        %(prog)s China my-data.linked
    `.

    To retain any mentions in one input that are mentions in another, use: `
        # field 4 is link prediction; . means match any non-empty text
        %(prog)s '.' --field 4 --aux other-data.linked my-data.linked
    `.

    To retain only LOC entity mentions in the input
        # field 3 is NER IOB tag, but CoNLL delimits by space
        %(prog)s LOC --field 3 --aux <(tr ' ' '\\t' < conll03/tags.eng) my-data.linked
    `.

    All tokens in the auxiliary file must align with the input, with documents
    delimited by lines beginning '-DOCSTART-' or '-X-'. Other lines have fields
    delimited by tabs.
    """

    def __init__(self, system, expr, aux=None, field=None, ignore_case=False, debug=False):
        if aux is None and field is not None:
            raise ValueError('--field requires --aux to be set')
        self.system = system
        self.expr = expr
        self.aux = aux
        self.field = field
        self.ignore_case = ignore_case
        self.debug = debug

    def __call__(self):
        out_file = io.BytesIO()
        writer = Writer(out_file)
        string_matches = re.compile(self.expr).search
        if self.aux is not None:
            aux_reader = _read_aux(open(self.aux))

            if self.field is None:
                field_slice = slice(None, None)
            else:
                field_slice = slice(self.field - 1, self.field)
        else:
            aux_reader = None
            aux_doc = None
            field_slice = None

        for doc in Reader(open(self.system)):
            if aux_reader:
                aux_doc = next(aux_reader)
                assert len(aux_doc) == doc.n_tokens, 'Expected same number of tokens, got {} in aux and {} in input'.format(len(aux_doc), doc.n_tokens)

            for sent, ment in self.filter_mentions(string_matches, doc,
                                                   aux_doc, field_slice):
                sent.explode_mention(ment)

            writer.write(doc)
        return out_file.getvalue()

    def filter_mentions(self, string_matches, doc, aux_doc=None, field_slice=None):
        """Generates mentions that don't match the criterion"""
        for sentence in doc.sentences:
            for span in sentence.spans:
                if not isinstance(span, Mention):
                    continue
                mention = span
                if aux_doc is None:
                    text = mention.text
                else:
                    aux_mention = aux_doc[mention.start:mention.end]
                    transposed = list(itertools.izip_longest(*aux_mention))[field_slice]
                    text = '\n'.join(' '.join(tup) for tup in transposed)
                if self.debug:
                    print(mention, repr(text), sep='\t', file=sys.stderr)

                if not string_matches(text):
                    yield sentence, mention

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('grep', help=cls.__doc__.split('\n')[0],
                          description=textwrap.dedent(cls.__doc__.split('\n', 1)[1].rstrip()) or None,
                          formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument('expr', help='A PCRE regular expression to match against mention metadata')
        p.add_argument('system', metavar='FILE')
        p.add_argument('--aux', help='Aligned text to match within')
        p.add_argument('--debug', action='store_true', default=False, help='Show text being matched against on stderr')
        p.add_argument('-f', '--field', type=int, default=None,
                       help='tabs-delimited field in the auxiliary file to match against (default any)')
        p.add_argument('-i', '--ignore-case', action='store_true', default=False)
        p.set_defaults(cls=cls)
        return p
