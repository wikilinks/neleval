~~~~~~~
neleval
~~~~~~~

Python command-line evaluation scripts for `TAC <http://tac.nist.gov>`__ entity
linking and related wikification, named entity disambiguation, and within- and
cross-document coreference tasks.

|version| |licence| |py-versions|

|issues| |build| |docs| |coverage|


It aims for **fast** and **flexible** coreference resolution and
**sophisticated** named entity recognition evaluation, such as partial scores
for partial overlap between gold and system mentions. CEAF, in particular, is
much faster to calculate here than in the `CoNLL-11/12 scorer
<https://github.com/conll/reference-coreference-scorers>`_. It boasts features
such as :ref:`configurable metrics <measures>`; accounting for or ignoring
cross-document coreference (see the :ref:`evaluate --by-doc
<command_evaluate_by_doc>` flag); plotting to compare evaluation by system,
measure and corpus subset; and bootstrap-based confidence interval calculation
for document-wise evaluation metrics.

Documentation
.............

.. toctree::
    :maxdepth: 2

    install
    usage
    measures
    formats
    cli
    scripts

References
..........

This project extends the work described in:

* Ben Hachey, Joel Nothman and Will Radford (2014),
  `"Cheap and easy entity evaluation" <http://acl2014.org/acl2014/P14-2/pdf/P14-2076.pdf>`_.
  In Proceedings of ACL.

It was used as the official scorer for Entity (Discovery and) Linking in 2014--:

* Heng Ji, Joel Nothman and Ben Hachey (2014),
  `"Overview of TAC-KBP2014 Entity Discovery and Linking Tasks" <http://nlp.cs.rpi.edu/paper/edl2014overview.pdf>`_,
  In Proceedings of the Text Analysis Conference.
* Heng Ji, Joel Nothman, Ben Hachey and Radu Florian (2015),
  `"Overview of TAC-KBP2015 Tri-lingual Entity Discovery and Linking Tasks" <http://nlp.cs.rpi.edu/paper/kbp2015.pdf>`_,
  In Proceedings of the Text Analysis Conference.
* Heng Ji and Joel Nothman (2016),
  `"Overview of TAC-KBP2016 Tri-lingual EDL and Its Impact on End-to-End KBP" <http://nlp.cs.rpi.edu/paper/kbp2016.pdf>`_,
  In Proceedings of the Text Analysis Conference.

Changelog
.........

.. include:: ../CHANGELOG.rst


.. |py-versions| image:: https://img.shields.io/pypi/pyversions/neleval.svg
    :alt: Python versions supported

.. |version| image:: https://badge.fury.io/py/neleval.svg
    :alt: Latest version on PyPi
    :target: https://badge.fury.io/py/neleval

.. |build| image:: https://travis-ci.org/wikilinks/neleval.svg?branch=master
    :alt: Travis CI build status
    :target: https://travis-ci.org/wikilinks/neleval

.. |issues| image:: https://img.shields.io/github/issues/wikilinks/neleval.svg
    :alt: Issue tracker
    :target: https://github.com/wikilinks/neleval

.. |coverage| image:: https://coveralls.io/repos/github/wikilinks/neleval/badge.svg
    :alt: Test coverage
    :target: https://coveralls.io/github/wikilinks/neleval

.. |docs| image:: https://readthedocs.org/projects/neleval/badge/?version=latest
     :alt: Documentation Status
     :target: https://neleval.readthedocs.io/en/latest/?badge=latest

.. |licence| image:: https://img.shields.io/badge/Licence-Apache%202.0-blue.svg
     :target: https://opensource.org/licenses/Apache-2.0
