#!/usr/bin/env python
from __future__ import print_function

import argparse
import textwrap
import re
import sys

#from .prepare import Prepare
from .evaluate import Evaluate
from .analyze import Analyze
from .significance import Significance, Confidence
#from .formats import Unstitch, Stitch, Tagme
#from .fetch_map import FetchMapping
#from .filter import FilterMentions
#from .rcv import ReutersCodes
from .tac import PrepareTac
from .configs import ListMeasures
from .summary import CompareMeasures, PlotSystems, ComposeMeasures

APPS = [
    Evaluate,
    ListMeasures,
    Analyze,
    Significance,
    Confidence,
    #Prepare,
    #FilterMentions,
    #Unstitch,
    #Stitch,
    #Tagme,
    #FetchMapping,
    #ReutersCodes,
    PrepareTac,
    CompareMeasures,
    PlotSystems,
    ComposeMeasures,
]


def main(args=sys.argv[1:]):
    p = argparse.ArgumentParser(prog='neleval',
                                description='Evaluation tools for Named Entity Linking output.')
    sp = p.add_subparsers()
    subparsers = {}
    for cls in APPS:
        hyphened_name = re.sub('([A-Z])', r'-\1', cls.__name__).lstrip('-').lower()
        help_text = cls.__doc__.split('\n')[0]
        desc = textwrap.dedent(cls.__doc__.rstrip())

        csp = sp.add_parser(hyphened_name,
                            help=help_text,
                            description=desc,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        cls.add_arguments(csp)
        subparsers[cls] = csp

    namespace = vars(p.parse_args(args))
    cls = namespace.pop('cls')
    try:
        obj = cls(**namespace)
    except ValueError as e:
        subparsers[cls].error(e.message)
    result = obj()
    if result is not None:
        print(result)

if __name__ == '__main__':
    main()
