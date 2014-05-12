from __future__ import division

from functools import partial

import numpy as np

from .munkres import linear_assignment

# Clusters are most readily evaluated when represented as a
# mapping from cluster ID to set of mention IDs. Could equally
# use a set of frozensets, but lose debugging info.


def cluster_sim_f1(a, b):
    """

    "Entity-based" measure in CoNLL; #4 in CEAF paper
    """
    if a and b:
        return len(a & b) / (len(a) + len(b))
    return 0.


def cluster_sim_overlap(a, b):
    """Intersection of sets

    "Mention-based" measure in CoNLL; #3 in CEAF paper
    """
    return len(a & b)


def ceaf(true, pred, similarity=cluster_sim_f1):
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
    return p, r, 2 * p * r / (p + r)


entity_ceaf = partial(ceaf, similarity=cluster_sim_f1)
mention_ceaf = partial(ceaf, similarity=cluster_sim_overlap)
