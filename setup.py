from __future__ import print_function
import sys
from setuptools import setup


def read_markdown(path):
    try:
        import pypandoc
    except ImportError:
        if 'sdist' in sys.argv:
            print('WARNING: Could not import pypandoc to convert README.md to RST!',
                  file=sys.stderr)
        return open(path).read()
    return pypandoc.convert(path, 'rst')


setup(name='neleval',
      version='3.0.0',
      download_url='https://github.com/wikilinks/neleval/tree/v3.0.0',

      description='Evaluation utilities for named entity linking and (cross-document) coreference resolution',
      author='Joel Nothman, Ben Hachey, Will Radford',
      author_email='joel.nothman+neleval@gmail.com',
      packages=['neleval'],
      url='https://github.com/wikilinks/neleval',
      keywords=['clustering', 'coreference', 'evaluation', 'entity disambiguation'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering',
      ],
      licence='Apache 2.0',

      long_description=read_markdown('README.md'),

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
