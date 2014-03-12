import re
import sys

def log(*s):
    print >> sys.stderr, s

WIKI_PREFIX = re.compile('^http://[^.]+.wikipedia.org/wiki/')
def normalise_link(l):
    """ Normalise links.
    * strips Wikipedia article prefixes.
    * replaces spaces with underscores.
    """
    return WIKI_PREFIX.sub('', l).replace(' ', '_')
