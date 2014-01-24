from distutils.core import setup
setup(name='conll03_nel_eval',
      version='0.1',
      description='Evaluation utilities for the CoNLL03 Named Entity Linking corpus',
      py_modules=['conll03_nel_eval'],
      url='https://github.com/benhachey/conll03_nel_eval',
      entry_points = {
        'console_scripts': [
            'conll03_nel_eval = conll03_nel_eval.__main__:main',
        ],
      }
)
