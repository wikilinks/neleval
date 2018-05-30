from collections import defaultdict
import json

from .document import Reader
from .utils import utf8_open


class SelectAlternatives(object):
    """Handle KB ambiguity in the gold standard by modifying it to match system

    The following back-off strategy applies for each span with gold standard
    ambiguity:

        * attempt to match it to the top candidate for that span
        * attempt to match it to the top candidate for any span in that
          document
        * attempt to match it to the top candidate for any span in the
          collection
        * default to select the first listed candidate

    The altered gold standard will be output.
    """

    def __init__(self, system, gold, fields='eid'):
        self.system = system
        self.gold = gold
        self.fields = fields.split(',') if fields != '*' else '*'

    def _get_key(self, candidate):
        if self.fields == '*':
            # avoid comparing on score
            return (candidate.eid, candidate.__dict__)
        return tuple(getattr(candidate, field, None)
                     for field in self.fields)

    def __call__(self):
        system = self.system
        if not isinstance(system, list):
            system = Reader(utf8_open(system))
        gold = self.gold
        if not isinstance(gold, list):
            gold = Reader(utf8_open(gold))

        # XXX: maybe we should choose most frequent rather than any
        by_span = {}
        by_doc = defaultdict(set)
        by_collection = set()
        for doc in system:
            for ann in doc.annotations:
                key = self._get_key(ann.candidates[0])
                by_span[ann] = key
                by_doc[ann.docid].add(key)
                by_collection.add(key)

        by_doc.default_factory = None

        out = []
        for doc in gold:
            for ann in doc.annotations:
                if len(ann.candidates) <= 1:
                    out.append(str(ann))
                    continue

                keys = [self._get_key(cand) for cand in ann.candidates]
                try:
                    matched = keys.index(by_span[ann.span])
                except (KeyError, IndexError):
                    # span not annotated or candidate not matched
                    try:
                        doc_keys = by_doc[ann.docid]
                    except KeyError:
                        # no system annotations in this doc
                        doc_keys = set()
                    collection_match = None
                    for i, key in enumerate(keys):
                        if key in doc_keys:
                            matched = i
                            break
                        if collection_match is None and key in by_collection:
                            collection_match = i
                    else:
                        if collection_match is None:
                            matched = 0
                        else:
                            matched = collection_match

                ann.candidates = [ann.candidates[matched]]
                out.append(str(ann))

        return '\n'.join(out)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('-f', '--fields', default='eid',
                       help='Comma-delimited list of fields to match '
                            'candidates at the same span between system and '
                            'gold. "*" will require match on all fields; '
                            'default is "eid".')
        p.add_argument('-g', '--gold', required=True,
                       help='Path to gold standard annotations')
        p.add_argument('system', metavar='FILE',
                       help='Path to system annotations')
        p.set_defaults(cls=cls)
        return p


class WeightsForHierarchy(object):
    """Translate a hierarchy of types into a sparse matrix of type-pair weights

    Input is a JSON object mapping parents to children in the hierarchy.
    Output is a three-column TSV with:

        * gold type
        * system type
        * weight

    The weights are assigned such that where the system type is an ancestor of
    the gold type with d edges between them, it will score (decay ** d).
    """

    def __init__(self, hierarchy_json, decay=0.5):
        self.hierarchy_json = hierarchy_json
        self.decay = decay
        if decay > 1.0 or decay < 0:
            raise ValueError('Decay must be greater than 0 and at most 1')

    def __call__(self):
        if self.hierarchy_json.startswith('{'):
            hierarchy = json.loads(self.hierarchy_json)
        else:
            with open(self.hierarchy_json) as f:
                hierarchy = json.load(f)

        out = []
        for parent, children in hierarchy.items():
            self._descend(out, hierarchy, parent, children, self.decay)
        # XXX: returning this all as a string rather than writing to a stream
        #      is yuck!
        return '\n'.join('%s\t%s\t%f' % tup for tup in out)

    def _descend(self, out, hierarchy, gold, children, weight):
        for child in children:
            out.append((gold, child, weight))
            self._descend(out, hierarchy, gold, hierarchy.get(child, ()),
                          weight * self.decay)

    @classmethod
    def add_arguments(cls, p):
        p.add_argument('-d', '--decay', default=0.5, type=float,
                       help='Decay value for systems selecting an ancestor '
                       'of the gold type')
        p.add_argument('hierarchy_json', metavar='FILE',
                       help='Path to hierarchy JSON')
        p.set_defaults(cls=cls)
        return p
