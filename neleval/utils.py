import re
import logging
import sys
import json
from io import open

log = logging.getLogger()

WIKI_PREFIX = re.compile('^http://[^.]+.wikipedia.org/wiki/')


try:
    unicode = unicode
except NameError:
    unicode = str


def normalise_link(l):
    """ Normalise links.
    * strips Wikipedia article prefixes.
    * replaces spaces with underscores.
    """
    return WIKI_PREFIX.sub('', l).replace(' ', '_')


def utf8_open(path, mode='r'):
    return open(path, mode, encoding='utf8')


def json_dumps(obj, sort_keys=True, indent=4):
    def default(o):
        if 'numpy' in sys.modules and \
           isinstance(o, sys.modules['numpy'].integer):
            return int(o)
        raise TypeError
    return json.dumps(obj, sort_keys=sort_keys, indent=indent, default=default)
