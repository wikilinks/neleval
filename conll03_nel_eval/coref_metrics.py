from __future__ import division, print_function

from functools import partial
from collections import defaultdict
import itertools
import re
import os
import subprocess
import tempfile
import warnings

import numpy as np

from .munkres import linear_assignment


# TODO: Blanc and standard clustering metrics (e.g. http://scikit-learn.org/stable/modules/clustering.html)
# TODO: cite originating papers
# XXX: perhaps use set (or list) of sets rather than dict of sets


######## Debug mode comparison to reference implementation ########


def _get_reference_coref_scorer_path():
    path = os.environ.get('COREFSCORER', None)
    if path is None:
        return None
    if os.path.isdir(path):
        path = os.path.join(path, 'scorer.pl')
    if not os.path.isfile(path):
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
                         Coreference:\s
                         Recall:\s
                         \(([^/]+)/([^)]+)\)
                         .*
                         Precision:\s
                         \(([^/]+)/([^)]+)\)
                         ''',
                         section,
                         re.DOTALL | re.VERBOSE)
        r_num, r_den, p_num, p_den = map(float, match.groups())
        stats = _prf(p_num, p_den, r_num, r_den)
        if one_metric:
            return stats
        else:
            metric = section[:section.index(':')]
            res[metric] = stats
    return res


def _run_reference_coref_scorer(true, pred, metric='all',
                                script=REFERENCE_COREF_SCORER_PATH):
    true_file = tempfile.NamedTemporaryFile(prefix='coreftrue', delete=False)
    pred_file = tempfile.NamedTemporaryFile(prefix='corefpred', delete=False)
    write_conll_coref(true, pred, true_file, pred_file)
    true_file.close()
    pred_file.close()
    output = subprocess.check_output([script, metric, true_file.name,
                                      pred_file.name])
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

        def wrapper(true, pred):
            our_results = fn(true, pred)
            ref_results = _run_reference_coref_scorer(true, pred, metric)
            assert len(our_results) == len(ref_results) == 3
            for our_val, ref_val, name in zip(our_results, ref_results, 'PRF'):
                if abs(our_val - ref_val) > 1e-3:
                    msg = 'Our {}={}; reference {}={}'.format(name, our_val,
                                                              name, ref_val)
                    raise AssertionError(msg)
            return our_results
        return wrapper
    return decorator


######## Data formats ########

def mapping_to_sets(mapping):
    """
    >>> sets = mapping_to_sets({'a': 1, 'b': 2, 'c': 1}).items()
    >>> sorted((k, sorted(v)) for k, v in sets)
    [(1, ['a', 'c']), (2, ['b'])]
    """
    s = defaultdict(set)
    for m, k in mapping.items():
        s[k].add(m)
    return dict(s)


def sets_to_mapping(s):
    """
    >>> sorted(sets_to_mapping({1: {'a', 'c'}, 2: {'b'}}).items())
    [('a', 1), ('b', 2), ('c', 1)]
    """
    return {m: k for k, ms in s.items() for m in ms}


def read_conll_coref(f):
    res = defaultdict(set)
    # TODO: handle annotations over document boundary
    i = 0
    opened = {}
    for l in f:
        if l.startswith('#'):
            continue
        l = l.split()
        if not l:
            assert not opened
            continue

        i += 1
        tag = l[-1]

        for match in re.finditer(r'\(?[0-9]+\)?', tag):
            match = match.group()
            cid = match.strip('()')
            if match.startswith('('):
                assert cid not in opened
                opened[cid] = i
            if match.endswith(')'):
                res[cid].add((opened.pop(cid), i))
    return dict(res)


def write_conll_coref(true, pred, true_file, pred_file):
    """Artificially aligns mentions as CoNLL coreference data
    """
    # relabel clusters
    true = {'({})'.format(i + 1): s for i, s in enumerate(true.values())}
    pred = {'({})'.format(i + 1): s for i, s in enumerate(pred.values())}
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
    p = p_num / p_den if p_den > 0 else 0. # TODO default 0 or 1?
    r = r_num / r_den if r_den > 0 else 0. # TODO default 0 or 1?
    return p, r, _f1(p, r)


######## Cluster comparison ########

def dice(a, b):
    """

    "Entity-based" measure in CoNLL; #4 in CEAF paper
    """
    if a and b:
        return len(a & b) / (len(a) + len(b))
    return 0.


def overlap(a, b):
    """Intersection of sets

    "Mention-based" measure in CoNLL; #3 in CEAF paper
    """
    return len(a & b)


######## Coreference metrics ########


def ceaf(true, pred, similarity=dice):
    """

    >>> true = {'A': {1,2,3,4,5}, 'B': {6,7}, 'C': {8, 9, 10, 11, 12}}
    >>> pred_a = {'A': {1,2,3,4,5}, 'B': {6,7, 8, 9, 10, 11, 12}}
    >>> pred_b = {'A': {1,2,3,4,5,8, 9, 10, 11, 12}, 'B': {6,7}}
    >>> pred_c = {'A': {1,2,3,4,5, 6,7, 8, 9, 10, 11, 12}}
    >>> pred_d = {i: {i,} for i in range(1, 13)}
    >>> mention_ceaf(true, pred_a)[-1]  # doctest: +ELLIPSIS
    0.83...
    >>> entity_ceaf(true, pred_a)[-1]  # doctest: +ELLIPSIS
    0.73...
    >>> mention_ceaf(true, pred_b)[-1]  # doctest: +ELLIPSIS
    0.58...
    >>> entity_ceaf(true, pred_b)[-1]  # doctest: +ELLIPSIS
    0.66...
    >>> mention_ceaf(true, pred_c)  # doctest: +ELLIPSIS
    (0.416..., 0.416..., 0.416...)
    >>> entity_ceaf(true, pred_c)  # doctest: +ELLIPSIS
    (0.588..., 0.196..., 0.294...)
    >>> mention_ceaf(true, pred_d)  # doctest: +ELLIPSIS
    (0.25, 0.25, 0.25)
    >>> entity_ceaf(true, pred_d)  # doctest: +ELLIPSIS
    (0.111..., 0.444..., 0.177...)
    """
    X = np.empty((len(true), len(pred)))
    pred = list(pred.values())
    for R, Xrow in zip(true.values(), X):
        Xrow[:] = [similarity(R, S) for S in pred]
    indices = linear_assignment(-X)

    numerator = sum(X[indices[:, 0], indices[:, 1]])
    true_denom = sum(similarity(R, R) for R in true.values())
    pred_denom = sum(similarity(S, S) for S in pred)
    #p = numerator / pred_denom # TODO redundant?
    #r = numerator / true_denom # TODO redundant?
    #return _prf(numerator, pred_denom, numerator, true_denom)
    return int(numerator), int(pred_denom-numerator), int(true_denom-numerator) # TODO ok?


@_cross_check('ceafe')
def entity_ceaf(true, pred):
    # TODO ok?
    return ceaf(true, pred, similarity=dice)


@_cross_check('ceafm')
def mention_ceaf(true, pred):
    # TOTO ok?
    return ceaf(true, pred, similarity=overlap)


# TODO remove if above ok
#entity_ceaf = _cross_check('ceafe')(partial(ceaf, similarity=dice))
#mention_ceaf = _cross_check('ceafm')(partial(ceaf, similarity=overlap))


def _b_cubed(A, B, A_mapping, B_mapping, EMPTY=frozenset([])):
    res = 0.
    for m, k in A_mapping.items():
        A_cluster = A.get(k, EMPTY)
        res += len(A_cluster & B.get(B_mapping.get(m), EMPTY)) / len(A_cluster)
    return res, len(A_mapping)


@_cross_check('bcub')
def b_cubed(true, pred):
    """

    TODO: tests
    """
    true_mapping = sets_to_mapping(true)
    pred_mapping = sets_to_mapping(pred)
    p_num, p_den = _b_cubed(pred, true, pred_mapping, true_mapping)
    r_num, r_den = _b_cubed(true, pred, true_mapping, pred_mapping)
    return _prf(p_num, p_den, r_num, r_den) # TODO handle diff p & r tps


def pairwise_f1_old(true, pred):
    """Measure the proportion of correctly identified pairwise coindexations

    TODO: tests
    """
    pred_mapping = sets_to_mapping(pred)
    correct = 0
    for cluster in true.values():
        for m1, m2 in itertools.combinations(cluster, 2):
            if pred_mapping.get(m1) == pred_mapping.get(m2):
                correct += 1
    p_den = sum(len(cluster) * (len(cluster) - 1) for cluster in pred.values()) * 2
    r_den = sum(len(cluster) * (len(cluster) - 1) for cluster in true.values()) * 2
    #return _prf(correct, p_den, correct, r_den)
    return int(correct), int(p_den-correct), int(r_den-correct) # TODO ok for tp, fp, fn?

def _pairs(C):
    "Return pairs of instances across all clusters in C"
    return frozenset(itertools.chain(*[itertools.combinations(c,2) for c in C]))

def _matrix(true, pred):
    "Return (tp, fp, fn) tuple for true and predicted sets"
    i = true & pred
    return len(i), len(pred)-len(i), len(true)-len(i)

def pairwise_f1(true, pred):
    return _matrix(_pairs(true.values()), _pairs(pred.values()))


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

    Examples from Vilain et al. (1995):
    >>> muc({1: {'A', 'B', 'C', 'D'}},
    ...     {1: {'A', 'B'}, 2: {'C', 'D'}})  # doctest: +ELLIPSIS
    (1.0, 0.66..., 0.8)
    >>> muc({1: {'A', 'B'}, 2: {'C', 'D'}},
    ...     {1: {'A', 'B', 'C', 'D'}})  # doctest: +ELLIPSIS
    (0.66..., 1.0, 0.8)
    >>> muc({1: {'A', 'B', 'C'}}, {1: {'A', 'C'}})  # doctest: +ELLIPSIS
    (1.0, 0.5, 0.66...)
    >>> muc({1: {'B', 'C', 'D', 'E', 'G', 'H', 'J'}},
    ...     {1: {'A', 'B', 'C'}, 2: {'D', 'E', 'F'}, 3: {'G', 'H', 'I'}})
    ... # doctest: +ELLIPSIS
    (0.5, 0.5, 0.5)
    >>> muc({1: {'A', 'B', 'C'}, 2: {'D', 'E', 'F', 'G'}},
    ...     {1: {'A', 'B'}, 2: {'C', 'D'}, 3: {'F', 'G', 'H'}})
    ... # doctest: +ELLIPSIS
    (0.5, 0.4, 0.44...)
    """
    p_num, p_den = _vilain(pred, sets_to_mapping(true))
    r_num, r_den = _vilain(true, sets_to_mapping(pred))
    return _prf(p_num, p_den, r_num, r_den) # TODO handle diff p & r tps


# Configuration constants
ALL_CMATCHES = 'all'
TAC_CMATCHES = 'tac'
TMP_CMATCHES = 'tmp'
NO_CMATCHES = 'none'
CMATCH_SETS = {
    ALL_CMATCHES: [
        mention_ceaf,
        entity_ceaf,
        b_cubed,
        pairwise_f1,
        muc,
        ],
    TAC_CMATCHES: [
        mention_ceaf,
        b_cubed,
        ],
    TMP_CMATCHES: [
        mention_ceaf,
        entity_ceaf,
        pairwise_f1,
        ],
    NO_CMATCHES: [],
}
DEFAULT_CMATCH_SET = TMP_CMATCHES # TODO until matrix and tp, fp, fn updates



if REFERENCE_COREF_SCORER_PATH is not None:
    if _run_reference_coref_scorer({}, {}).get('bcub') != (0., 0., 0.):
        warnings.warn('Not using coreference metric debug mode:'
                      'The script is producing invalid output')
        REFERENCE_COREF_SCORER_PATH = None


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='CoNLL2011-2 coreference evaluator')
    ap.add_argument('key_file', type=argparse.FileType('r'))
    ap.add_argument('response_file', type=argparse.FileType('r'))
    args = ap.parse_args()
    METRICS = {
        'bcubed': b_cubed,
        'ceafe': entity_ceaf,
        'ceafm': mention_ceaf,
        'muc': muc,
        'pairs': pairwise_f1,
    }
    key = read_conll_coref(args.key_file)
    response = read_conll_coref(args.response_file)
    print('Metric', 'P', 'R', 'F1', sep='\t')
    for name, fn in sorted(METRICS.items()):
        print(name, *('{:0.2f}'.format(100 * x) for x in fn(key, response)), sep='\t')
