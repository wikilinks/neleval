import sys
import argparse
import io

from .coref_metrics import read_conll_coref
from .annotation import Annotation, Candidate
from .utils import log, unicode


def _coref_to_annotations(clusters, docid, with_kb=False, cross_doc=False):
    if cross_doc:
        nil_refmt = u'%s'
    else:
        nil_refmt = u'%%s:%s' % docid

    for cid, mentions in clusters.items():
        if not with_kb:
            # prefix all with NIL
            cid = u'NIL' + unicode(cid)
        if cid.startswith(u'NIL'):
            cid = nil_refmt % cid

        for (start, end) in mentions:
            yield Annotation(docid, start, end, [Candidate(cid)])


class PrepareConllCoref(object):
    "Import format from CoNLL 2011-2 coreference shared task for evaluation"
    def __init__(self, system, with_kb=False, cross_doc=False):
        self.system = system
        self.with_kb = with_kb
        self.cross_doc = cross_doc

    BEGIN = '#begin document '
    END = '#end document'

    def _iter_conll_annotations(self):
        doc_no = 0
        docid = None
        buf = []
        for l in self.system:
            if l.startswith(self.BEGIN):
                assert not ''.join(buf).strip()
                assert docid is None
                docid = '_'.join(l.strip()[len(self.BEGIN):].split())
                if not docid:
                    docid = 'doc%d' % doc_no
                if isinstance(docid, bytes):
                    docid = docid.decode('utf8')

            elif l.startswith(self.END):
                assert docid is not None
                log.debug("Read %d lines from %r" % (len(buf), docid))
                buf = ''.join(buf)
                if isinstance(buf, bytes):
                    buf = buf.decode('utf8')
                # TODO: check start/end offsets agree in definition with EDL!
                clusters = read_conll_coref(io.StringIO(buf))
                annotations = list(_coref_to_annotations(clusters, docid, self.with_kb, self.cross_doc))
                log.debug("> Yielded %d annotations" % len(annotations))
                for annot in annotations:
                    yield annot
                docid = None
                doc_no += 1
                buf = []

            elif l.startswith('#'):
                continue

            else:
                buf.append(l)

    def __call__(self):
        return u'\n'.join(unicode(a)
                          for a in sorted(self._iter_conll_annotations()))

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('system', nargs='?', default=sys.stdin, type=argparse.FileType('r'))
        p.add_argument('--with-kb', default=False, action='store_true', help='By default all cluster labels are treated as NILs. This flag treats all as KB IDs unless prefixed by "NIL"')
        p.add_argument('--cross-doc', default=False, action='store_true', help='By default, label space is independent per document. This flag assumes global label space.')
        p.set_defaults(cls=cls)
        return p
