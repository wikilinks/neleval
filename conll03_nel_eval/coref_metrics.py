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
        stats = p_num, p_den, r_num, r_den
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
            res = fn(true, pred)
            our_results = _prf(*res)
            ref_results = _prf(*_run_reference_coref_scorer(true, pred, metric))
            assert len(our_results) == len(ref_results) == 3
            for our_val, ref_val, name in zip(our_results, ref_results, 'PRF'):
                if abs(our_val - ref_val) > 1e-3:
                    msg = 'Our {}={}; reference {}={}'.format(name, our_val,
                                                              name, ref_val)
                    raise AssertionError(msg)
            return res
        return wrapper
    return decorator


######## Data formats ########

def mapping_to_sets(mapping):
    """
    Input: {cluster_item: cluster_name} dictionary
    Output: {cluster_name: set([cluster_items])} dictionary
    """
    s = defaultdict(set)
    for m, k in mapping.items():
        s[k].add(m)
    return dict(s)


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
    p = p_num / p_den if p_den > 0 else 0.
    r = r_num / r_den if r_den > 0 else 0.
    return p, r, _f1(p, r)

def _to_matrix(p_num, p_den, r_num, r_den):
    ptp = p_num
    fp = p_den - p_num
    rtp = r_num
    fn = r_den - r_num
    return ptp, fp, rtp, fn


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
    "Luo (2005). On coreference resolution performance metrics. In EMNLP."
    X = np.empty((len(true), len(pred)))
    pred = list(pred.values())
    for R, Xrow in zip(true.values(), X):
        Xrow[:] = [similarity(R, S) for S in pred]
    indices = linear_assignment(-X)

    p_num = r_num = sum(X[indices[:, 0], indices[:, 1]])
    p_den = sum(similarity(R, R) for R in true.values())
    r_den = sum(similarity(S, S) for S in pred)
    return p_num, p_den, r_num, r_den


@_cross_check('ceafm')
def mention_ceaf(true, pred):
    "Luo (2005) phi_3"
    return ceaf(true, pred, similarity=overlap)


@_cross_check('ceafe')
def entity_ceaf(true, pred):
    "Luo (2005) phi_4"
    return ceaf(true, pred, similarity=dice)


def _b_cubed(A, B, A_mapping, B_mapping, EMPTY=frozenset([])):
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
    true_mapping = sets_to_mapping(true)
    pred_mapping = sets_to_mapping(pred)
    p_num, p_den = _b_cubed(pred, true, pred_mapping, true_mapping)
    r_num, r_den = _b_cubed(true, pred, true_mapping, pred_mapping)
    return p_num, p_den, r_num, r_den


def _pairs(C):
    "Return pairs of instances across all clusters in C"
    return frozenset(itertools.chain(
            *[itertools.combinations_with_replacement(c,2) for c in C]))

def _pairwise_f1(true, pred):
    "Return numerators and denominators for precision and recall"
    p_num = r_num = len(true & pred)
    p_den = len(pred)
    r_den = len(true)
    return p_num, p_den, r_num, r_den

def pairwise_f1(true, pred):
    "Return p_num, p_den, r_num, r_den over item pairs."
    return _pairwise_f1(_pairs(true.values()), _pairs(pred.values()))


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


# Configuration constants
ALL_CMATCHES = 'all'
MUC_CMATCHES = 'muc'
LUO_CMATCHES = 'luo'
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
    MUC_CMATCHES: [
        muc,
        ],
    LUO_CMATCHES: [
        muc,
        b_cubed,
        mention_ceaf,
        entity_ceaf,
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
DEFAULT_CMATCH_SET = ALL_CMATCHES



if REFERENCE_COREF_SCORER_PATH is not None:
    if _run_reference_coref_scorer({}, {}).get('bcub') != (0, 0, 0, 0):
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
        print(name, *('{:0.2f}'.format(100 * x) for x in _prf(*fn(key, response))), sep='\t')
