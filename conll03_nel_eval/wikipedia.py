#!/usr/bin/env python
"""
A small Wikipedia API wrapper.
"""
import urllib
import json
from .utils import log

API_URL = 'http://en.wikipedia.org/w/api.php'

class Wikipedia(object):
    def redirected(self, title):
        """Return outgoing redirect or title."""
        params = {
            'action': 'query',
            'prop': 'info',
            'format': 'json',
            'titles': self._utf8(title),
            'redirects': 'true',
            }
        response = self._fetch(params)
        for redirect in self._redirects(response):
            if 'tofragment' in redirect:
                continue # redirect points to an article section
            return redirect['to']
        return title

    def redirects(self, title):
        """Yield incoming redirect list or []."""
        params = {
            'action': 'query',
            'generator': 'backlinks',
            'gbllimit': 'max',
            'gblfilterredir': 'redirects',
            'gblnamespace': '0',
            'gbltitle': self._utf8(title),
            'prop': 'info',
            'redirects': 'true',
            }
        response = self._fetch(params)
        for redirect in self._redirects(response):
            if redirect.get('to') != title:
                continue # redirect does not point to title
            if 'tofragment' in redirect:
                continue # redirect points to an article section
            yield redirect['from']

    def _redirects(self, r):
        """Return redirects list from redirects json response."""
        return r.get('query',{}).get('redirects',[])

    # GENERAL UTILITIES
    def _utf8(self, s):
        """Yield back utf8-encoded versions of s."""
        if type(s) is unicode:
            return s.encode('utf8')
        else:
            return s

    def _fetch(self, params):
        """Return json-formatted Wikipedia API result for given parameters."""
        params['format'] = 'json'
        url = API_URL + '?' + urllib.urlencode(params)
        log.info('Fetching {}'.format(url))
        return json.loads(urllib.urlopen(url).read())
