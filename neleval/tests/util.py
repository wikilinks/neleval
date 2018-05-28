
def check_correct(expected, actual):
    assert set(expected.keys()) <= set(actual.keys()), 'Different keys\nexpected\t{}\nactual\t{}'.format(sorted(expected.keys()), sorted(actual.keys()))
    for k in expected:
        exp = expected[k]
        act = actual[k]
        if hasattr(exp, '__iter__') and not hasattr(exp, 'items'):
            assert exp == act, 'Different on key "{}".\nexpected\t{}\nactual\t{}'.format(k, exp, act)
        else:
            assert exp == act, 'Different on key "{}".\nexpected\t{}\nactual\t{}'.format(k, exp, act)
    return True
