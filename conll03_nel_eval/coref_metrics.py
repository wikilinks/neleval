from __future__ import division

from functools import partial
from collections import defaultdict
import itertools

import numpy as np

from .munkres import linear_assignment


# TODO: Blanc and standard clustering metrics (e.g. http://scikit-learn.org/stable/modules/clustering.html)
# TODO: cite originating papers
# XXX: perhaps use set (or list) of sets rather than dict of sets


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


def _f1(a, b):
    return 2 * a * b / (a + b)


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
    p = numerator / pred_denom
    r = numerator / true_denom
    return p, r, _f1(p, r)


entity_ceaf = partial(ceaf, similarity=dice)
mention_ceaf = partial(ceaf, similarity=overlap)


def _b_cubed(A, B, A_mapping, B_mapping, EMPTY=frozenset([])):
    res = 0.
    for m, k in A_mapping.items():
        A_cluster = A.get(k, EMPTY)
        res += len(A_cluster & B.get(B_mapping.get(m), EMPTY)) / len(A_cluster)
    res /= len(A_mapping)
    return res


def b_cubed(true, pred):
    """

    TODO: tests
    """
    true_mapping = sets_to_mapping(true)
    pred_mapping = sets_to_mapping(pred)
    p = _b_cubed(pred, true, pred_mapping, true_mapping)
    r = _b_cubed(true, pred, true_mapping, pred_mapping)
    return p, r, _f1(p, r)


def pairwise_f1(true, pred):
    """Measure the proportion of correctly identified pairwise coindexations

    This is called MUC score, and erroneously cited to Vilain et al. (2102) in
    the CoNLL 2011-2012 Shared Task descriptions.

    TODO: tests
    """
    pred_mapping = sets_to_mapping(pred)
    correct = 0
    for cluster in true.values():
        for m1, m2 in itertools.combinations(cluster, 2):
            if pred_mapping.get(m1) == pred_mapping.get(m2):
                correct += 1
    p = correct / sum(len(cluster) - 1 for cluster in pred.values())
    r = correct / sum(len(cluster) - 1 for cluster in true.values())
    return p, r, _f1(p, r)


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
    return numerator / denominator


def vilain(true, pred):
    """The MUC evaluation metric defined in Vilain et al. (1995)

    This calculates recall error for each true cluster C as the number of
    response clusters that would need to be merged in order to produce a
    superset of C.

    Examples from Vilain et al. (1995):
    >>> vilain({1: {'A', 'B', 'C', 'D'}},
    ...        {1: {'A', 'B'}, 2: {'C', 'D'}})  # doctest: +ELLIPSIS
    (1.0, 0.66..., 0.8)
    >>> vilain({1: {'A', 'B'}, 2: {'C', 'D'}},
    ...        {1: {'A', 'B', 'C', 'D'}})  # doctest: +ELLIPSIS
    (0.66..., 1.0, 0.8)
    >>> vilain({1: {'A', 'B', 'C'}}, {1: {'A', 'C'}})  # doctest: +ELLIPSIS
    (1.0, 0.5, 0.66...)
    >>> vilain({1: {'B', 'C', 'D', 'E', 'G', 'H', 'J'}},
    ...        {1: {'A', 'B', 'C'}, 2: {'D', 'E', 'F'}, 3: {'G', 'H', 'I'}})
    ... # doctest: +ELLIPSIS
    (0.5, 0.5, 0.5)
    >>> vilain({1: {'A', 'B', 'C'}, 2: {'D', 'E', 'F', 'G'}},
    ...        {1: {'A', 'B'}, 2: {'C', 'D'}, 3: {'F', 'G', 'H'}})
    ... # doctest: +ELLIPSIS
    (0.5, 0.4, 0.44...)
    """
    p = _vilain(pred, sets_to_mapping(true))
    r = _vilain(true, sets_to_mapping(pred))
    return p, r, _f1(p, r)
