Entity linking evaluation
=========================

Python command-line evaluation scripts for `TAC <http://www.nist.gov/tac/>`__
entity linking and related wikification, named entity disambiguation, and
within- and cross-document coreference tasks.

|version| |licence| |py-versions|

|issues| |build| |docs| |coverage|

It aims for **fast** and **flexible** coreference resolution and
**sophisticated** named entity recognition evaluation, such as partial scores
for partial overlap between gold and system mentions. CEAF, in particular, is
much faster to calculate here than in the `CoNLL-11/12 scorer
<https://github.com/conll/reference-coreference-scorers>`__. It boasts features
such as configurable metrics; accounting for or ignoring cross-document
coreference (see the ``evaluate --by-doc`` flag); plotting to compare
evaluation by system, measure and corpus subset; and bootstrap-based confidence
interval calculation for document-wise evaluation metrics.

Requires that ``python`` (2.7, with Py3k support experimental/partial)
be installed on your system with ``numpy`` (and preferably ``scipy`` for
fast CEAF calculation) and ``joblib``. ``matplotlib`` is required for
the ``plot-systems`` command.

See a list of commands with:

.. code:: bash

    ./nel --help

Or install onto your Python path (e.g. with
``pip install git+https://github.com/wikilinks/neleval``) then

.. code:: bash

    python -m neleval --help

TAC-KBP 2014 EDL quickstart
===========================

.. code:: bash

    ./scripts/run_tac14_evaluation.sh \
        /path/to/gold.xml \              # TAC14 gold standard queries/mentions
        /path/to/gold.tab \              # TAC14 gold standard link and nil annotations
        /system/output/directory \       # directory containing (only) TAC14 system output files
        /script/output/directory \       # directory to which results are written
        number_of_jobs                   # number of jobs for parallel mode

Each file in in the system output directory is scored against gold.tab.

Similar facility is available for TAC-KBP'15 EDL.

More details
============

See `the documentation <https://neleval.readthedocs.io>`__ for more
details.


.. |py-versions| image:: https://img.shields.io/pypi/pyversions/neleval.svg
    :alt: Python versions supported

.. |version| image:: https://badge.fury.io/py/neleval.svg
    :alt: Latest version on PyPi
    :target: https://badge.fury.io/py/neleval

.. |build| image:: https://travis-ci.org/wikilinks/neleval.svg?branch=master
    :alt: Travis CI build status
    :scale: 100%
    :target: https://travis-ci.org/wikilinks/neleval

.. |issues| image:: https://img.shields.io/github/issues/wikilinks/neleval.svg
    :alt: Issue tracker
    :target: https://github.com/wikilinks/neleval

.. |coverage| image:: https://coveralls.io/repos/github/wikilinks/neleval/badge.svg
    :alt: Test coverage
    :target: https://coveralls.io/github/wikilinks/neleval

.. |docs| image:: https://readthedocs.org/projects/neleval/badge/?version=latest
     :alt: Documentation Status
     :scale: 100%
     :target: https://neleval.readthedocs.io/en/latest/?badge=latest

.. |licence| image:: https://img.shields.io/badge/Licence-Apache%202.0-blue.svg
     :target: https://opensource.org/licenses/Apache-2.0
