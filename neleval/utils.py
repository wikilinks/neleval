import re
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
log = logging.getLogger()

WIKI_PREFIX = re.compile('^http://[^.]+.wikipedia.org/wiki/')


def normalise_link(l):
    """ Normalise links.
    * strips Wikipedia article prefixes.
    * replaces spaces with underscores.
    """
    return WIKI_PREFIX.sub('', l).replace(' ', '_')
