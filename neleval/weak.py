#!/usr/bin/env python
from .annotation import Annotation
from .document import ENC
from .utils import unicode

class ToWeak(object):
    """Convert annotations to char-level for weak evaluation

    A better approach is to use measures with partial overlap support.
    """
    def __init__(self, fname):
        self.fname = fname

    def __call__(self):
        return u'\n'.join(unicode(a) for a in self.annotations()).encode(ENC)

    def annotations(self):
        for line in open(self.fname):
            a = Annotation.from_string(line.rstrip('\n').decode(ENC))
            for i in range(a.start, a.end+1):
                yield Annotation(a.docid, i, i+1, a.candidates)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('fname', metavar='FILE')
        p.set_defaults(cls=cls)
        return p
