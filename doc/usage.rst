Basic Usage
~~~~~~~~~~~

The NEL evaluation tools are invoked using ``neleval``, or ``./nel`` inside the
repository. Usage:

.. code:: bash

    neleval <command> [<args>]

To list available commands:

.. code:: bash

    neleval

To get help for a specific command:

.. code:: bash

    neleval <command> -h

See :ref:`cli`.

The commands that are relevant to `TAC KBP entity
linking <http://nlp.cs.rpi.edu/kbp/2014/>`__ evaluation and analysis are
described below.

Basic usage
===========

The following describes a typical workflow. See also :ref:`scripts`.

Convert gold standard to evaluation format
------------------------------------------

For data in `TAC14 format <format_tac14>`__:

.. code:: bash

    neleval prepare-tac \
        -q /path/to/gold.xml \    # gold queries/mentions file
        /path/to/gold.tab \       # gold KB/NIL annotations file
        > gold.combined.tsv

For data in TAC12 and TAC13 format, remove extra columns first, e.g.:

.. code:: bash

    cat /path/to/gold.tab \
        | cut -f1,2,3 \
        > gold.tab
    neleval prepare-tac \
        -q /path/to/gold.xml \
        gold.tab \
        > gold.combined.tsv

Convert system output to evaluation format
------------------------------------------

For data in `TAC14 format <format_tac14>`__:

.. code:: bash

    neleval prepare-tac \
        -q /path/to/system.xml \  # system mentions file
        /path/to/system.tab \     # system KB/NIL annotations
        > system.combined.tsv

For data in TAC12 and TAC13 format, add dummy NE type column first,
e.g.:

.. code:: bash

    cat /path/to/system.tab \
        | awk 'BEGIN{OFS="\t"} {print $1,$2,"NA",$3}' \
        > system.tab
    neleval prepare-tac \
        -q /path/to/gold.xml \    # gold queries/mentions file
        system.tab \              # system KB/NIL annotations
        > system.combined.tsv

Evaluate system output
----------------------

To calculate micro-averaged scores for all evaluation measures:

.. code:: bash

    neleval evaluate \
        -m all \                  # report all evaluation measures
        -f tab \                  # print results in tab-separated format
        -g gold.combined.tsv \    # prepared gold standard annotation
        system.combined.tsv \     # prepared system output
        > system.evaluation

To list available evaluation measures:

.. code:: bash

    neleval list-measures

Advanced usage
==============

The following describes additional commands for analysis. See also
`run\_tac14\_all.sh <../tree/master/scripts/run_tac14_all.sh>`__ (TODO)
and `run\_tac13\_all.sh <../tree/master/scripts/run_tac13_all.sh>`__.

Calculate confidence intervals
------------------------------

To calculate confidence intervals using bootstrap resampling:

.. code:: bash

    neleval confidence \
        -m strong_typed_link_match \ # report CI for TAC14 wikification measure
        -f tab \                  # print results in tab-separated format
        -g gold.combined.tsv \    # prepared gold standard annotation
        system.combined.tsv \     # prepared system output
        > system.confidence

We recommend that you ``pip install joblib`` and use ``-j NUM_JOBS`` to
run this in parallel. This is also faster if an individual evaluation
measure is specified (e.g., strong\_typed\_link\_match) rather than
groups of measures (e.g., tac).

The
`run\_report\_confidence.sh <../tree/master/scripts/run_report_confidence.sh>`__
script is available to create reports comparing multiple systems.

Note that bootstrap resampling is not appropriate for nil clustering
measures. For more detail, see `the Significance wiki
page <Significance>`__.

Calculate significant differences
---------------------------------

It is also possible to calculate pairwise differences:

.. code:: bash

    neleval significance \
        --permute \               # use permutation method
        -f tab \                  # print results in tab-separated format
        -g gold.combined.tsv \    # prepared gold standard annotation
        system1.combined.tsv \    # prepared system1 output
        system2.combined.tsv \    # prepared system2 output
        > system1-system2.significance

We recommend calculating significance for selected system pairs as it
can take a while over all N choose 2 combinations of systems. You can
also use ``-j NUM_JOBS`` to run this in parallel.

Note that bootstrap resampling is not appropriate for nil clustering
measures. For more detail, see `the Significance wiki
page <Significance>`__.

Analyze error types
-------------------

To create a table of classification errors:

.. code:: bash

    neleval analyze \
        -s \                      # print summary table
        -g gold.combined.tsv \    # prepared gold standard annnotation
        system.combined.tsv \     # prepared system output
        > system.analysis

Without the ``-s`` flag, the ``analyze`` command will list and
categorize differences between the gold standard and system output.

Filter data for evaluation on subsets
=====================================

The following describes a workflow for evaluation over subsets of
mentions. See also
`run\_tac14\_filtered.sh <../tree/master/scripts/run_tac14_filtered.sh>`__
(TODO) and
`run\_tac13\_filtered.sh <../tree/master/scripts/run_tac13_filtered.sh>`__.

Filter prepared data
--------------------

Prepared data is in a simple tab-separated format with one mention per
line and six columns: ``document_id``, ``start_offset``, ``end_offset``,
``kb_or_nil_id``, ``score``, ``entity_type``. It is possible to use
command line tools (e.g., ``grep``, ``awk``) to select mentions for
evaluation, e.g.:

.. code:: bash

    cat gold.combined.tsv \       # prepared gold standard annotation
        | egrep "^eng-(NG|WL)-" \ # select newsgroup and blog (WB) mentions
        > gold.WB.tsv             # filtered gold standard annotation
    cat system.combined.tsv \     # prepared system output
        | egrep "^eng-(NG|WL)-" \ # select newsgroup and blog (WB) mentions
        > system.WB.tsv           # filtered system output

Evaluate on filtered data
-------------------------

After filtering, evaluation is run as before:

.. code:: bash

    neleval evaluate \
        -m all \                  # report all evaluation measures
        -f tab \                  # print results in tab-separated format
        -g gold.WB.tsv \          # filtered gold standard annotation
        system.WB.tsv \           # filtered system output
        > system.WB.evaluation

Evaluate each document or entity type
-------------------------------------

To get a score for each document, or each entity type, as well as the
macro-averaged score across documents, use ``--group-by`` in
:ref:`command_evaluate`. See :ref:`grouped_measures`.
