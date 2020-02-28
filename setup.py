from __future__ import print_function
import sys
import os
from setuptools import setup


def read_markdown(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        if 'sdist' in sys.argv:
            print('WARNING: did not find %r' % filename, file=sys.stderro)
        return
    try:
        import pypandoc
    except ImportError:
        if 'sdist' in sys.argv:
            print('WARNING: Could not import pypandoc to convert README.md to RST!',
                  file=sys.stderr)
        return open(path).read()
    return pypandoc.convert(path, 'rst')


VERSION = '3.1.1'

setup(name='neleval',
      version=VERSION,
      download_url='https://github.com/wikilinks/neleval/tree/v' + VERSION,

      description='Command-line evaluation tools for named entity linking and (cross-document) coreference resolution',
      author='Joel Nothman, Ben Hachey, Will Radford',
      author_email='joel.nothman+neleval@gmail.com',
      packages=['neleval'],
      url='https://github.com/wikilinks/neleval',
      keywords=['clustering', 'coreference', 'evaluation', 'entity disambiguation'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
      ],
      licence='Apache 2.0',

      long_description=open('README.rst').read(),

      entry_points={
          'console_scripts': [
              'neleval = neleval.__main__:main',
          ],
      },
      install_requires=[
          'numpy',
      ],
      extras_require={
          'significance': [
              'joblib',
          ],
          'ceaf': [
              'scipy',
          ],
          'plots': [
              'matplotlib',
          ],
          'dev': [
              'pyflakes',
              'nose',
          ],
      }
)
