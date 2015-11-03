from __future__ import division, print_function

from collections import defaultdict, OrderedDict
import itertools
import heapq
import warnings

import numpy as np
from scipy import sparse

from .coref_metrics import sets_to_mapping, sets_to_matrices, overlap, _disjoint_max_assignment
from .utils import log


def summarise(true, pred, candidates):
    print('%d true clusters' % len(true))
    print('%d pred clusters' % len(pred))
    print('%d candidates' % len(candidates))
    g_candidates = defaultdict(list)
    s_candidates = defaultdict(list)
    for g_m, s_m in candidates:
        g_candidates[g_m].append(s_m)
        s_candidates[s_m].append(g_m)
    n_1to1 = 0
    n_1tomany = 0
    n_manyto1 = 0
    for g_m, s_ms in g_candidates.items():
        if len(s_ms) == 1:
            if len(s_candidates.get(s_ms[0])) == 1:
                n_1to1 += 1
            else:
                n_1tomany += 1
    for s_m, g_ms in s_candidates.items():
        if len(g_ms) == 1:
            if len(g_candidates.get(g_ms[0])) == 1:
                pass
            else:
                n_manyto1 += 1
    print('%d 1to1' % n_1to1)
    print('%d 1 gold to many sys' % n_1tomany)
    print('%d 1 sys to many gold' % n_manyto1)
    print('%d many to many' % (len(candidates) - n_1to1 - n_1tomany - n_manyto1))


def fix_unaligned(true, pred, candidates, similarity=overlap,
                  method='max-assignment', n_iter=None):
    # candidates is [((true_mention_id, cluster), (pred_mention_id, cluster))]
    # for each true mention require only one pred mention per cluster id and v.v.
    if method == 'summary':
        return summarise(true, pred, candidates)

    log.info('Preparing for %s with %dx%d custers and '
             '%d unaligned pairs', method, len(true),
             len(pred), len(candidates))
    if method == 'max-assignment':
        method = _max_assignment
    elif method == 'greedy':
        method = _greedy
    else:
        raise ValueError('Unknown method: %r' % method)

    true = OrderedDict(true)
    pred = OrderedDict(pred)
    true_to_ind = {label: i for i, label in enumerate(true)}
    pred_to_ind = {label: i for i, label in enumerate(pred)}
    candidates = [((m_true, true_to_ind[l_true]),
                   (m_pred, pred_to_ind[l_pred]))
                  for (m_true, l_true), (m_pred, l_pred) in candidates]
    by_cluster_pair = defaultdict(list)
    for (m_true, l_true), (m_pred, l_pred) in candidates:
        by_cluster_pair[l_true, l_pred].append((m_true, m_pred))

    true, pred, _ = sets_to_matrices(true, pred)
    if similarity is not overlap:
        # For dice, need appropriate norms in modifications to X
        raise NotImplementedError
    X = similarity.vectorized(true, pred)
    l_true, l_pred, n_mentions = zip(*((l_true, l_pred, len(mentions))
                                       for (l_true, l_pred), mentions
                                       in by_cluster_pair.items()))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
        X[l_true, l_pred] += n_mentions

    fixes = method(X, n_iter, by_cluster_pair,
                   norm_func=lambda l_true, l_pred: 1)
    log.info('Done fixing')
    return fixes


def _match(X, by_cluster_pair, norm_func, method_str,
           fixes, true_match, pred_match, zero_on_del=False):
    # Delete candidates for matched clusters
    X[true_match, pred_match] = 0
    true_fixed = set()
    pred_fixed = set()
    for l_true, l_pred in zip(true_match, pred_match):
        for m_true, m_pred in by_cluster_pair.pop((l_true, l_pred), ()):
            true_fixed.add(m_true)
            pred_fixed.add(m_pred)
            fixes.append((method_str, m_true, m_pred))
    # Delete candidates for matched mentions
    true_match = set(true_match)
    pred_match = set(pred_match)
    for (l_true, l_pred), mention_pairs in by_cluster_pair.items():
        if l_true in true_match or l_pred in pred_match:
            diff = len(mention_pairs)
            mention_pairs = [(m_true, m_pred)
                             for m_true, m_pred in mention_pairs
                             if m_true not in true_fixed
                             and m_pred not in pred_fixed]
            by_cluster_pair[l_true, l_pred] = mention_pairs
            diff -= len(mention_pairs)
            X[l_true, l_pred] -= diff / norm_func(l_true, l_pred)
            if not mention_pairs:
                del by_cluster_pair[l_true, l_pred]
                if zero_on_del:
                    # Only keep entries with candidates
                    X[l_true, l_pred] = 0


def _max_assignment(X, n_iter, by_cluster_pair, norm_func):
    max_n_iter = min(X.shape)
    if n_iter is None:
        n_iter = max_n_iter
    fixes = []
    for i in range(min(n_iter, max_n_iter)):
        log.info('Maximising, iteration %d, nnz=%d', i, X.nnz)
        _, trues, preds = _disjoint_max_assignment(X, return_mapping=True)
        _match(X, by_cluster_pair, norm_func, 'RMA iter %d' % (n_iter + 1),
               fixes, trues, preds)
        X.eliminate_zeros()
        if not by_cluster_pair:
            break
    return fixes


def _greedy(X, n_iter, by_cluster_pair, norm_func):
    all_true, all_pred = zip(*by_cluster_pair.keys())
    # Retain only cells with candidates
    X2 = sparse.csr_matrix(X.shape, dtype=X.dtype)
    X2[all_true, all_pred] = X[all_true, all_pred].A.ravel()
    X = X2

    max_n_iter = X.shape[0]
    if n_iter is None:
        n_iter = max_n_iter
    fixes = []
    for i in range(min(n_iter, max_n_iter)):
        if i % 50 == 0 and i:
            X.eliminate_zeros()
        if i % 100 == 0:
            log.info('Greedy, iteration %d, nnz=%d', i, X.nnz)
        idx = np.argmax(X.data)  # makes this O(n^2); could be done with heap
        if X.data[idx] == 0:
            break
        l_true = np.searchsorted(X.indptr, idx, 'right') - 1
        l_pred = X.indices[idx]
        assert (l_true, l_pred) in by_cluster_pair
        _match(X, by_cluster_pair, norm_func, 'Greedy iter %d' % (n_iter + 1),
               fixes, [l_true], [l_pred], zero_on_del=True)
        if not by_cluster_pair:
            break
    return fixes
