class Evaluate(object):
    def __init__(self, fname, gold):
        print 'Evaluating', fname, gold

    @classmethod
    def add_arguments(cls, sp):
        p = sp.add_parser('evaluate', help='Evaluate system output')
        p.add_argument('-g', '--gold')
        p.set_defaults(cls=cls)
        return p
