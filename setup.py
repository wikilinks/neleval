from setuptools import setup
setup(name='neleval',
      version='0.2',
      description='Evaluation utilities named entity linking and cross-document coreference',
      packages=['neleval'],
      url='https://github.com/wikilinks/neleval',
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
          'dev': [
            'pyflakes',
            'nose',
          ],
      }
)
