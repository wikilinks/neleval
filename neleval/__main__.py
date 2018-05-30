#!/usr/bin/env python
from __future__ import print_function

import argparse
import textwrap
import re
import sys
import traceback
import logging

from .prepare import SelectAlternatives, WeightsForHierarchy
from .evaluate import Evaluate
from .analyze import Analyze
from .significance import Significance, Confidence
from .tac import PrepareTac, PrepareTac15
from .brat import PrepareBrat
from .import_ import PrepareConllCoref
from .configs import ListMeasures
from .summary import CompareMeasures, PlotSystems, ComposeMeasures, RankSystems
from .weak import ToWeak
from .document import ValidateSpans

APPS = [
    Evaluate,
    ValidateSpans,
    ListMeasures,
    Analyze,
    Significance,
    Confidence,
    PrepareTac,
    PrepareTac15,
    PrepareBrat,
    PrepareConllCoref,
    CompareMeasures,
    RankSystems,
    PlotSystems,
    ComposeMeasures,
    ToWeak,
    SelectAlternatives,
    WeightsForHierarchy,
]


def main(args=sys.argv[1:]):
    p = argparse.ArgumentParser(prog='neleval',
                                description='Evaluation tools for Named Entity Linking output.')
    p.add_argument('--verbose', dest='log_level', action='store_const',
                   const=logging.DEBUG, default=logging.INFO)
    p.add_argument('--quiet', dest='log_level', action='store_const',
                   const=logging.ERROR)
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
    logging.basicConfig(level=namespace.pop('log_level'), format='%(levelname)s\t%(asctime)s\t%(message)s')
    try:
        cls = namespace.pop('cls')
    except KeyError:
        p.print_help()
        return
    try:
        obj = cls(**namespace)
    except ValueError as e:
        subparsers[cls].error(str(e) + "\n" + traceback.format_exc())
    result = obj()
    if result is not None:
        print(result)

if __name__ == '__main__':
    main()
