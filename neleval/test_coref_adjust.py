from .coref_adjust import fix_unaligned
from nose.tools import assert_list_equal, assert_equal


def assert_fixes_equal(actual, expected):
    fixes = {tup[1:] for tup in actual}
    assert_list_equal(sorted(fixes), sorted(expected))


def fix_unaligned_rev_args(true, pred, cands, *args, **kwargs):
    fixes = fix_unaligned(pred, true, [(y, x) for x, y in cands], *args, **kwargs)
    return [(n, y, x) for n, x, y in fixes]


def test_maximising():
    true = {'a': {1, 2}, 'b': {3}}
    pred = {'A': {1, 4}}
    cands = [((2, 'a'), (4, 'A')),
             ((3, 'a'), (4, 'A'))]
    for method in ['max-assignment', 'single-best']:
        assert_fixes_equal(fix_unaligned(true, pred, cands, method=method), [(2, 4)])
        assert_fixes_equal(fix_unaligned_rev_args(true, pred, cands, method=method), [(2, 4)])
    assert_list_equal(fix_unaligned(true, pred, cands, method='unambiguous'), [])
    assert_list_equal(fix_unaligned_rev_args(true, pred, cands, method='unambiguous'), [])

###    true = {'a': {1, 2, 3}, 'b': {4, 5}, 'c': {6}}
###    pred = {'A': {7, 8, 9}, 'B': {10, 11}, 'C': {12}}
###    cands = [()]

###    # TODO: Shuffling all arbitrary labels should get the same result where result is deterministic


###[[1, 1, 1],
### [0, 1, 1],
### [0, 0, 1]]



def test_no_double_fix():
    true = {'a': {1,}}
    pred = {'A': {2, 3}}
    cands = [((1, 'a'), (2, 'A')),
             ((1, 'a'), (3, 'A')),
             ]
    for method in ['max-assignment', 'single-best']:
        assert_equal(len(fix_unaligned(true, pred, cands, method=method)), 1)
        assert_equal(len(fix_unaligned_rev_args(true, pred, cands, method=method)), 1)


def test_repeated_maximising():
    pass


def test_mention_epsilon():
    true = {'a': {1, 3}, 'b': {2}}
    pred = {'A': {4, 5}, 'B': {3}}
    cands1 = [((1, 'a'), (4, 'A')),
              ((1, 'a'), (5, 'A')),
              ((2, 'b'), (4, 'A'))]
    cands2 = [((1, 'a'), (4, 'A')),
              ((1, 'a'), (5, 'A')),
              ((2, 'b'), (5, 'A'))]
    assert_fixes_equal(fix_unaligned(true, pred, cands1, method='single-best', n_iter=1), [(1, 5)])
    assert_fixes_equal(fix_unaligned(true, pred, cands2, method='single-best', n_iter=1), [(1, 4)])

    # Force max assignment for b in first iteration to be B
    true = {'a': {1, 3}, 'b': {2, 7, 8, 9}}
    pred = {'A': {4, 5}, 'B': {3, 7, 8, 9}}
    for method in ['single-best', 'max-assignment']:
        assert_fixes_equal(fix_unaligned(true, pred, cands1, method=method), [(1, 5), (2, 4)])
        assert_fixes_equal(fix_unaligned(true, pred, cands2, method=method), [(1, 4), (2, 5)])
