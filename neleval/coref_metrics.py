from __future__ import division, print_function

from collections import defaultdict
import itertools
import operator
import re
import os
import subprocess
import tempfile
import warnings
import time
import functools
import array
import signal


try:
    from scipy import sparse
    from scipy.sparse import csgraph
except ImportError:
    sparse = None

try:
    import numpy as np
    from .munkres import linear_assignment
except ImportError:
    np = None

try:   # Py3k
    range = xrange
    zip = itertools.izip
    values = dict.viewvalues
    keys = dict.viewkeys
except NameError:
    values = dict.values
    keys = dict.keys


class TimeoutError(Exception):
    pass


class timeout:
    def __init__(self, seconds=1):
        self.seconds = seconds

    def handle_timeout(self, signum, frame):
        raise TimeoutError()

    def __enter__(self):
        try:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        except Exception:
            # I can't find documentation of what happens if executed off Unix
            pass

    def __exit__(self, type, value, traceback):
        try:
            signal.alarm(0)
        except Exception:
            # I can't find documentation of what happens if executed off Unix
            pass


# TODO: cite originating papers
# TODO: calculate all from contingency matrices and marginals


######## Debug mode comparison to reference implementation ########


def _get_reference_coref_scorer_path():
    path = os.environ.get('COREFSCORER', None)
    if path is None:
        return None
    if os.path.isdir(path):
        path = os.path.join(path, 'scorer.pl')
    if os.path.isfile(path):
        warnings.warn('Using coreference metric debug mode!')
    else:
        warnings.warn('Not using coreference metric debug mode:'
                      '{} is not a file'.format(path))
    return path


REFERENCE_COREF_SCORER_PATH = _get_reference_coref_scorer_path()


def _parse_reference_coref_scorer(output):
    sections = output.split('\nMETRIC ')
    if len(sections) > 1:
        sections = sections[1:]  # strip preamble
        one_metric = False
    else:
        one_metric = True

    res = {}
    for section in sections:
        match = re.match(r'''
                         .*
                         \s
                         Recall:\s
                         \(([^/]+)/([^)]+)\)
                         .*?
                         Precision:\s
                         \(([^/]+)/([^)]+)\)
                         ''',
                         section,
                         re.DOTALL | re.VERBOSE)
        r_num, r_den, p_num, p_den = map(float, match.groups())
        stats = p_num, p_den, r_num, r_den
        if one_metric:
            return stats
        else:
            metric = section[:section.index(':')]
            res[metric] = stats
    return res


def _run_reference_coref_scorer(true, pred, metric='all'):
    true_file = tempfile.NamedTemporaryFile(prefix='coreftrue', delete=False)
    pred_file = tempfile.NamedTemporaryFile(prefix='corefpred', delete=False)
    write_conll_coref(true, pred, true_file, pred_file)
    true_file.close()
    pred_file.close()
    #start = time.time()
    output = subprocess.check_output([REFERENCE_COREF_SCORER_PATH,
                                      metric, true_file.name,
                                      pred_file.name, 'none'])
    #their_time = time.time() - start
    #print('Ran perl scorer', metric, 'in ', their_time, file=sys.stderr)
    #print(output[-400:], file=sys.stderr)
    os.unlink(true_file.name)
    os.unlink(pred_file.name)
    return _parse_reference_coref_scorer(output)


def _cross_check(metric):
    """A wrapper that will assert our output matches reference implementation

    Applies only if the environment variable COREFSCORER points to the
    reference implementation.
    """
    def decorator(fn):
        if REFERENCE_COREF_SCORER_PATH is None:
            return fn

        @functools.wraps(fn)
        def wrapper(true, pred):
            #start = time.time()
            our_results = fn(true, pred)
            #our_time = time.time() - start
            #print('Ran our', metric, 'in ', our_time, file=sys.stderr)
            ref_results = _prf(*_run_reference_coref_scorer(true, pred, metric))

            for our_val, ref_val, name in zip(_prf(*our_results), ref_results, 'PRF'):
                if abs(our_val - ref_val) > 1e-3:
                    msg = 'Our {} {}={}; reference {}={}'.format(metric,
                                                                 name, our_val,
                                                                 name, ref_val)
                    raise AssertionError(msg)
            return our_results
        return wrapper
    return decorator


######## Utilities ########

def mapping_to_sets(mapping):
    """
    Input: {cluster_item: cluster_name} dictionary
    Output: {cluster_name: set([cluster_items])} dictionary
    """
    s = defaultdict(set)
    for m, k in mapping.items():
        s[k].add(m)
    s.default_factory = None  # disable defaulting
    return s


def sets_to_mapping(s):
    """
    Input: {cluster_name: set([cluster_items])} dictionary
    Output: {cluster_item: cluster_name} dictionary
    """
    return {m: k for k, ms in s.items() for m in ms}


def read_conll_coref(f):
    res = defaultdict(set)
    # TODO: handle annotations over document boundary
    i = 0
    opened = defaultdict(list)
    for l in f:
        if l.startswith('#'):
            continue
        l = l.split()
        if not l:
            assert not opened
            continue

        i += 1
        tag = l[-1]
        text = l[-2] if len(l) > 1 else ''

        closed_here = []
        for match in re.finditer(r'\(?[^()|]+\)?', tag):
            match = match.group()
            cid = match.strip('()')
            if match.startswith('('):
                opened[cid].append((text, i))
            if match.endswith(')'):
                start = opened[cid].pop()
                if not opened[cid]:
                    del opened[cid]
                #assert start_cid == cid, 'Found %r, expected %r at %d' % (cid, start_cid, i)
                closed_here.append((cid, start))

        # keep only one mention of those with identical spans
        for _, mentions in itertools.groupby(closed_here,
                                             operator.itemgetter(1)):
            cid, start = list(mentions)[-1]  # keep the outermost
            res[cid].add((start[1], i))

    assert not opened

    res.default_factory = None  # disable defaulting
    return res


def write_conll_coref(true, pred, true_file, pred_file):
    """Artificially aligns mentions as CoNLL coreference data
    """
    # relabel clusters
    true = {'({})'.format(i + 1): s for i, s in enumerate(values(true))}
    pred = {'({})'.format(i + 1): s for i, s in enumerate(values(pred))}
    # make lookups
    true_mapping = sets_to_mapping(true)
    pred_mapping = sets_to_mapping(pred)
    # headers
    print('#begin document (XX); part 000', file=true_file)
    print('#begin document (XX); part 000', file=pred_file)
    # print all mentions
    for mention in set(true_mapping).union(pred_mapping):
        print('XX', true_mapping.get(mention, '-'), file=true_file)
        print('XX', pred_mapping.get(mention, '-'), file=pred_file)
    # footers
    print('#end document', file=true_file)
    print('#end document', file=pred_file)


def _f1(a, b):
    if a + b:
        return 2 * a * b / (a + b)
    return 0.


def _prf(p_num, p_den, r_num, r_den):
    p = p_num / p_den if p_den > 0 else 0.
    r = r_num / r_den if r_den > 0 else 0.
    return p, r, _f1(p, r)


def twinless_adjustment(true, pred):
    """Adjusts predictions for differences in mentions

    Following Cai and Strube's (SIGDIAL'10) `sys` variants on B-cubed and CEAF.
    This produces a different true, pred pair for each of precision and recall
    calculation.

    Thus for precision:
        * twinless true mentions -> pred singletons
        * twinless pred singletons -> discard
        * twinless pred non-singletons -> true singletons
    For recall:
        * twinless true -> pred singletons
        * twinless pred -> discard

    Returns : p_true, p_pred, r_true, r_pred
    """
    true_mapping = sets_to_mapping(true)
    pred_mapping = sets_to_mapping(pred)

    # common: twinless true -> pred singletons
    twinless_true = set(true_mapping) - set(pred_mapping)
    for i, mention in enumerate(twinless_true):
        pred_mapping[mention] = ('twinless_true', i)

    # recall: twinless pred -> discard
    r_pred = mapping_to_sets({m: k
                              for m, k in pred_mapping.items()
                              if m in true_mapping})

    # precision: twinless pred singletons -> discard; non-singletons -> true
    for i, (m, k) in enumerate(list(pred_mapping.items())):
        if m in true_mapping:
            continue
        if len(pred[k]) > 1:
            true_mapping[m] = ('twinless_pred', i)
        else:
            del pred_mapping[m]

    p_true = mapping_to_sets(true_mapping)
    p_pred = mapping_to_sets(pred_mapping)

    return p_true, p_pred, true, r_pred


def sets_to_matrices(true, pred):
    if sparse is None:
        raise RuntimeError('Cannot vectorize without scipy')
    # TODO: perhaps cache vectorized `true`
    vocabulary = defaultdict(None)
    vocabulary.default_factory = vocabulary.__len__
    true_indptr = array.array('i', [0])
    true_indices = array.array('i')
    for true_cluster in values(true):
        for item in true_cluster:
            true_indices.append(vocabulary[item])
        true_indptr.append(len(vocabulary))

    pred_indptr = array.array('i', [0])
    pred_indices = array.array('i')
    for pred_cluster in values(pred):
        for item in pred_cluster:
            pred_indices.append(vocabulary[item])
        pred_indptr.append(len(pred_indices))

    true_data = np.ones(len(true_indices), dtype=int)
    true_matrix = sparse.csr_matrix((true_data, true_indices, true_indptr),
                                    shape=(len(true), len(vocabulary)))
    pred_data = np.ones(len(pred_indices), dtype=int)
    pred_matrix = sparse.csr_matrix((pred_data, pred_indices, pred_indptr),
                                    shape=(len(pred), len(vocabulary)))
    #true_matrix.check_format(full_check=True)
    #pred_matrix.check_format(full_check=True)
    return true_matrix, pred_matrix, vocabulary


######## Cluster comparison ########

def dice(a, b):
    """

    "Entity-based" measure in CoNLL; #4 in CEAF paper
    """
    if a and b:
        return (2 * len(a & b)) / (len(a) + len(b))
    return 0.


def _vectorized_dice(true_matrix, pred_matrix):
    overlap = _vectorized_overlap(true_matrix, pred_matrix).astype(float)

    # The following should be no-ops
    assert overlap.format == true_matrix.format == pred_matrix.format == 'csr'

    true_sizes = np.diff(true_matrix.indptr)
    pred_sizes = np.diff(pred_matrix.indptr)

    denom = np.repeat(true_sizes, np.diff(overlap.indptr))
    denom += pred_sizes.take(overlap.indices)
    overlap.data *= 2 / denom

    return overlap

dice.vectorized = _vectorized_dice


def overlap(a, b):
    """Intersection of sets

    "Mention-based" measure in CoNLL; #3 in CEAF paper
    """
    return len(a & b)


def _vectorized_overlap(true_matrix, pred_matrix):
    return true_matrix * pred_matrix.T

overlap.vectorized = _vectorized_overlap


######## Coreference metrics ########


class OptionalDependencyWarning(Warning):
    pass


def _disjoint_max_assignment(similarities):
    global sparse
    if sparse is None:
        if hasattr(similarities, 'toarray'):
            # Due to supporting scipy without connected_components
            similarities = similarities.toarray()
        start = time.time()
        indices = linear_assignment(-similarities)
        runtime = time.time() - start
        if runtime > 1:
            warnings.warn('The assignment step in CEAF took a long time. '
                          'We may be able to calculate it faster if you '
                          'install scipy.', OptionalDependencyWarning)
        return similarities[indices[:, 0], indices[:, 1]].sum()

    # form n*n adjacency matrix
    where_true, where_pred = similarities.nonzero()
    where_pred = where_pred + similarities.shape[0]
    n = sum(similarities.shape)
    A = sparse.coo_matrix((np.ones(len(where_true)), (where_true, where_pred)),
                          shape=(n, n))
    try:
        n_components, components = csgraph.connected_components(A, directed=False)
    except (AttributeError, TypeError):
        warnings.warn('Could not use scipy.sparse.csgraph.connected_components.'
                      'Please update your scipy installation. '
                      'Calculating max-score assignment the slow way.')
        # HACK!
        sparse = None
        return _disjoint_max_assignment(similarities)

    if hasattr(similarities, 'toarray'):
        # faster to work in dense
        similarities = similarities.toarray()
    total = 0
    for i in range(n_components):
        mask = components == i
        component_true = np.flatnonzero(mask[:similarities.shape[0]])
        component_pred = np.flatnonzero(mask[similarities.shape[0]:])
        component_sim = similarities[component_true, :][:, component_pred]
        if 0 in component_sim.shape:
            pass
        if component_sim.shape == (1, 1):
            total += component_sim[0, 0]
        else:
            indices = linear_assignment(-component_sim)
            total += component_sim[indices[:, 0], indices[:, 1]].sum()
    #assert total == similarities[tuple(linear_assignment(-similarities).T)].sum()
    return total


def ceaf(true, pred, similarity=dice):
    "Luo (2005). On coreference resolution performance metrics. In EMNLP."
    if np is None:
        warnings.warn('numpy is required to calculate CEAF. '
                      'Returning 0', OptionalDependencyWarning)
        return 0, 0, 0, 0

    try:
        with timeout(900):
            if sparse is None or not hasattr(similarity, 'vectorized'):
                X = np.empty((len(true), len(pred)))
                pred = list(values(pred))
                for R, Xrow in zip(values(true), X):
                    Xrow[:] = [similarity(R, S) for S in pred]

                p_num = r_num = _disjoint_max_assignment(X)
                r_den = sum(similarity(R, R) for R in values(true))
                p_den = sum(similarity(S, S) for S in pred)
            else:
                true, pred, _ = sets_to_matrices(true, pred)
                X = similarity.vectorized(true, pred)
                p_num = r_num = _disjoint_max_assignment(X)
                r_den = similarity.vectorized(true, true).sum()
                p_den = similarity.vectorized(pred, pred).sum()
    except TimeoutError:
        warnings.warn('timeout for CEAF!')
        return 0, 0, 0, 0

    return p_num, p_den, r_num, r_den


def cs_ceaf(true, pred, similarity=dice):
    """CEAF with twinless adjustment from Cai and Strube (2010)"""
    p_true, p_pred, r_true, r_pred = twinless_adjustment(true, pred)
    # XXX: there is probably a better way to do this
    p_num, p_den, _, _ = ceaf(p_true, p_pred, similarity)
    _, _, r_num, r_den = ceaf(r_true, r_pred, similarity)
    return p_num, p_den, r_num, r_den


@_cross_check('ceafm')
def mention_ceaf(true, pred):
    "Luo (2005) phi_3"
    return ceaf(true, pred, similarity=overlap)


@_cross_check('ceafe')
def entity_ceaf(true, pred):
    "Luo (2005) phi_4"
    return ceaf(true, pred, similarity=dice)


def mention_cs_ceaf(true, pred):
    return cs_ceaf(true, pred, similarity=overlap)


def entity_cs_ceaf(true, pred):
    return cs_ceaf(true, pred, similarity=dice)


def _b_cubed(A, B, EMPTY=frozenset([])):
    A_mapping = sets_to_mapping(A)
    B_mapping = sets_to_mapping(B)
    res = 0.
    for m, k in A_mapping.items():
        A_cluster = A.get(k, EMPTY)
        res += len(A_cluster & B.get(B_mapping.get(m), EMPTY)) / len(A_cluster)
    return res, len(A_mapping)


@_cross_check('bcub')
def b_cubed(true, pred):
    """
    Bagga and Baldwin (1998). Algorithms for scoring coreference chains.
    In LREC Linguistic Coreference Workshop.

    TODO: tests
    """
    p_num, p_den = _b_cubed(pred, true)
    r_num, r_den = _b_cubed(true, pred)
    return p_num, p_den, r_num, r_den


def cs_b_cubed(true, pred):
    """b_cubed with twinless adjustment from Cai and Strube (2010)"""
    p_true, p_pred, r_true, r_pred = twinless_adjustment(true, pred)
    p_num, p_den = _b_cubed(p_pred, p_true)
    r_num, r_den = _b_cubed(r_true, r_pred)
    return p_num, p_den, r_num, r_den


def _positive_pairs(C):
    "Return pairs of instances across all clusters in C"
    return frozenset(itertools.chain.from_iterable(
        itertools.combinations(sorted(c), 2) for c in C))


def _negative_pairs(C):
    return frozenset(tuple(sorted(item_pair))
                     for cluster_pair in itertools.combinations(C, 2)
                     for item_pair in itertools.product(*cluster_pair))


def _pairwise(true, pred):
    """Return numerators and denominators for precision and recall,
    as well as size of symmetric difference, used in negative pairwise."""
    p_num = r_num = len(true & pred)
    p_den = len(pred)
    r_den = len(true)
    return p_num, p_den, r_num, r_den


def pairwise(true, pred):
    """Return p_num, p_den, r_num, r_den over item pairs

    As used in calcualting BLANC (see Luo, Pradhan, Recasens and Hovy (2014).

    >>> pairwise({1: {'a', 'b', 'c'}, 2: {'d'}},
    ...         {1: {'b', 'c'}, 2: {'d', 'e'}})
    (1, 2, 1, 3)
    """
    # Slow in some cases, perhaps, but much less memory consumed
    return pairwise_slow(true, pred)
    return _pairwise(_positive_pairs(values(true)),
                     _positive_pairs(values(pred)))


def _triangle(n):
    return n * (n - 1) // 2


def pairwise_negative(true, pred):
    """Return p_num, p_den, r_num, r_den over noncoreferent item pairs

    As used in calcualting BLANC (see Luo, Pradhan, Recasens and Hovy (2014).

    >>> pairwise_negative({1: {'a', 'b', 'c'}, 2: {'d'}},
    ...                   {1: {'b', 'c'}, 2: {'d', 'e'}})
    (2, 4, 2, 3)
    """
    true_pairs = _positive_pairs(values(true))
    pred_pairs = _positive_pairs(values(pred))
    n_pos_agreements = len(true_pairs & pred_pairs)

    true_mapping = sets_to_mapping(true)
    pred_mapping = sets_to_mapping(pred)
    extra_mentions = keys(true_mapping) ^ keys(pred_mapping)
    disagreements = {p for p in true_pairs ^ pred_pairs
                     if p[0] not in extra_mentions
                     and p[1] not in extra_mentions}

    n_common_mentions = len(keys(true_mapping) & keys(pred_mapping))
    n_neg_agreements = (_triangle(n_common_mentions) - n_pos_agreements -
                        len(disagreements))

    # Total number of negatives in each of pred and true:
    p_den = _triangle(len(pred_mapping)) - len(pred_pairs)
    r_den = _triangle(len(true_mapping)) - len(true_pairs)

    return n_neg_agreements, p_den, n_neg_agreements, r_den


def pairwise_slow(true, pred):
    p_den = sum(_triangle(len(pred_cluster)) for pred_cluster in values(pred))
    r_den = sum(_triangle(len(true_cluster)) for true_cluster in values(true))
    numerator = sum(_triangle(len(true_cluster & pred_cluster))
                    for true_cluster in values(true)
                    for pred_cluster in values(pred))
    return numerator, p_den, numerator, r_den


def pairwise_negative_slow(true, pred):
    trues = [len(true_cluster) for true_cluster in values(true)]
    preds = [len(pred_cluster) for pred_cluster in values(pred)]
    intersections = [[len(true_cluster & pred_cluster)
                      for true_cluster in values(true)]
                     for pred_cluster in values(pred)]
    n_pred = sum(preds)
    n_true = sum(trues)
    p_den = sum(a * (n_pred - a) for a in preds) // 2
    r_den = sum(a * (n_true - a) for a in trues) // 2
    row_sums = [sum(row) for row in intersections]
    N = sum(row_sums)
    col_sums = [sum(col) for col in zip(*intersections)]
    assert N == sum(col_sums)
    num = sum(n * (N - row_sums[row_idx] - col_sums[col_idx] + n)
              for row_idx, row in enumerate(intersections)
              for col_idx, n in enumerate(row)) // 2
    return num, p_den, num, r_den


def _slow_pairwise_negative(true, pred):
    """For testing comparison"""
    return _pairwise(_negative_pairs(values(true)),
                     _negative_pairs(values(pred)))


def _vilain(A, B_mapping):
    numerator = 0
    denominator = 0
    for cluster in A.values():
        corresponding = set()
        n_unaligned = 0
        for m in cluster:
            if m not in B_mapping:
                n_unaligned += 1
            else:
                corresponding.add(B_mapping[m])
        numerator += len(cluster) - n_unaligned - len(corresponding)
        denominator += len(cluster) - 1
    return numerator, denominator


@_cross_check('muc')
def muc(true, pred):
    """The MUC evaluation metric defined in Vilain et al. (1995)

    This calculates recall error for each true cluster C as the number of
    response clusters that would need to be merged in order to produce a
    superset of C.
    """
    p_num, p_den = _vilain(pred, sets_to_mapping(true))
    r_num, r_den = _vilain(true, sets_to_mapping(pred))
    return p_num, p_den, r_num, r_den


COREF_METRICS = {f.__name__: f for f in [b_cubed, entity_ceaf, mention_ceaf,
                                         muc, pairwise, pairwise_negative]}


if REFERENCE_COREF_SCORER_PATH is not None:
    if _run_reference_coref_scorer({}, {}).get('bcub') != (0., 0., 0., 0.):
        warnings.warn('Not using coreference metric debug mode:'
                      'The script is producing invalid output')
        REFERENCE_COREF_SCORER_PATH = None


if __name__ == '__main__':
    import argparse
    descr = ('CoNLL2011-2 coreference evaluator: use neleval\'s '
             'prepare-conll-coref and evaluate commands for more control')
    ap = argparse.ArgumentParser(description=descr)
    ap.add_argument('key_file', type=argparse.FileType('r'))
    ap.add_argument('response_file', type=argparse.FileType('r'))
    args = ap.parse_args()
    METRICS = {
        'bcubed': b_cubed,
        'ceafe': entity_ceaf,
        'ceafm': mention_ceaf,
        'muc': muc,
        'pairs': pairwise,
        'negpairs': pairwise_negative,
    }
    key = read_conll_coref(args.key_file)
    response = read_conll_coref(args.response_file)
    print('Metric', 'P', 'R', 'F1', sep='\t')
    for name, fn in sorted(METRICS.items()):
        print(name, *('{:0.2f}'.format(100 * x)
                      for x in _prf(*fn(key, response))), sep='\t')
