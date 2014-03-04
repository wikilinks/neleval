#!/usr/bin/env python
import argparse
import sys
from .prepare import Prepare
from .evaluate import Evaluate
from .analyze import Analyze
from .significance import Significance
from .formats import Unstitch, Stitch, Tagme
from .fetch_map import FetchMapping
from .filter import FilterMentions

APPS = [
    Evaluate,
    Analyze,
    Significance,
    Prepare,
    FilterMentions,
    Unstitch,
    Stitch,
    Tagme,
    FetchMapping,
]


def main(args=sys.argv[1:]):
    p = argparse.ArgumentParser(description='Evaluation tools for Named Entity Linking output.')
    sp = p.add_subparsers()
    for cls in APPS:
        cls.add_arguments(sp)

    namespace = vars(p.parse_args(args))
    cls = namespace.pop('cls')
    try:
        obj = cls(**namespace)
    except ValueError as e:
        p.error(e.message)
    print obj()

if __name__ == '__main__':
    main()
