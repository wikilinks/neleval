
.. _detailed_measures:

Measures in detail
~~~~~~~~~~~~~~~~~~

This describes measures as listed by :ref:`command_list_measures`.

+-----------------------------------------+------------------+--------------+-----------------+
| Measure                                 | Key              | Filter       | Aggregator      |
+=========================================+==================+==============+=================+
| **Mention evaluation measures**         |                  |              |                 |
+-----------------------------------------+------------------+--------------+-----------------+
| *strong\_mention\_match*                | span             | NA           | sets            |
+-----------------------------------------+------------------+--------------+-----------------+
| *strong\_typed\_mention\_match*         | span,type        | NA           | sets            |
+-----------------------------------------+------------------+--------------+-----------------+
| *strong\_linked\_mention\_match*        | span             | is\_linked   | sets            |
+-----------------------------------------+------------------+--------------+-----------------+
| **Linking evaluation measures**         |                  |              |                 |
+-----------------------------------------+------------------+--------------+-----------------+
| *strong\_link\_match*                   | span,kbid        | is\_linked   | sets            |
+-----------------------------------------+------------------+--------------+-----------------+
| *strong\_nil\_match*                    | span             | is\_nil      | sets            |
+-----------------------------------------+------------------+--------------+-----------------+
| *strong\_all\_match*                    | span,kbid        | NA           | sets            |
+-----------------------------------------+------------------+--------------+-----------------+
| *strong\_typed\_link\_match*            | span,type,kbid   | is\_linked   | sets            |
+-----------------------------------------+------------------+--------------+-----------------+
| *strong\_typed\_nil\_match*             | span,type        | is\_nil      | sets            |
+-----------------------------------------+------------------+--------------+-----------------+
| *strong\_typed\_all\_match*             | span,type,kbid   | NA           | sets            |
+-----------------------------------------+------------------+--------------+-----------------+
| **Document-level tagging evaluation**   |                  |              |                 |
+-----------------------------------------+------------------+--------------+-----------------+
| *entity\_match*                         | docid,kbid       | is\_linked   | sets            |
+-----------------------------------------+------------------+--------------+-----------------+
| **Clustering evaluation measures**      |                  |              |                 |
+-----------------------------------------+------------------+--------------+-----------------+
| *muc*                                   | span             | NA           | muc             |
+-----------------------------------------+------------------+--------------+-----------------+
| *b\_cubed*                              | span             | NA           | b\_cubed        |
+-----------------------------------------+------------------+--------------+-----------------+
| *b\_cubed\_plus*                        | span,kbid        | NA           | b\_cubed        |
+-----------------------------------------+------------------+--------------+-----------------+
| *entity\_ceaf*                          | span             | NA           | entity\_ceaf    |
+-----------------------------------------+------------------+--------------+-----------------+
| *mention\_ceaf*                         | span             | NA           | mention\_ceaf   |
+-----------------------------------------+------------------+--------------+-----------------+
| *pairwise*                              | span             | NA           | pairwise        |
+-----------------------------------------+------------------+--------------+-----------------+

Custom measures
===============

A custom measure can be specified on the command-line as:

``<aggregator>:<filter>:<key>``

such as


``sets:None:span+kbid`` for *strong_all_match*

.. _grouped_measures:

Grouped measures
================

By default measures are aggregated over the corpus as a whole. Using the
``--by-doc`` and/or ``--by-type`` flags to :ref:`command_evaluate` will instead
aggregate measures per document or entity type, and then report
per-doc/type and overall (micro- and macro-averaged) performance. *Note
that micro-average does not equate to whole-corpus aggregation for
coreference aggregates, but represents clustering performance
disregarding cross-document coreference.*

.. _measure_key:

Key
===

The key defines how system output is matched against the gold standard.

+-----------+-------------------------------------------------------+
| Key       | Description                                           |
+===========+=======================================================+
| *docid*   | Document identifier must be the same                  |
+-----------+-------------------------------------------------------+
| *start*   | Start offset must be the same                         |
+-----------+-------------------------------------------------------+
| *end*     | End offset must be the same                           |
+-----------+-------------------------------------------------------+
| *span*    | Shorthand for (docid, start, end)                     |
+-----------+-------------------------------------------------------+
| *type*    | Entity type must be the same                          |
+-----------+-------------------------------------------------------+
| *kbid*    | KB identifier must be the same, or must both be NIL   |
+-----------+-------------------------------------------------------+

Filter
======

The filter defines what mentions are removed before precision, recall
and f-score calculations.

+-----------------------+----------------------------------------------------+
| Filter                | Description                                        |
+=======================+====================================================+
| *is\_linked*          | Only keep mentions that are resolved to known KB   |
|                       | identifiers                                        |
+-----------------------+----------------------------------------------------+
| *is\_nil*             | Only keep mentions that are not resolved to known  |
|                       | KB identifiers                                     |
+-----------------------+----------------------------------------------------+
| *is\_first*           | Only keep the first mention in a document of a     |
|                       | given KB/NIL identifier                            |
+-----------------------+----------------------------------------------------+

Note that the *is\_first* filter is intended to provide clustering
evaluation similar to the *entity\_match* evaluation of linking
performance.

.. _measure_aggregator:

Aggregator
==========

The aggregator defines how corpus-level scores are computed from
individual instances.

+------------------------------+----------------------------------------------------+
| Aggregator                   | Description                                        |
+==============================+====================================================+
| **Mention, linking,          |                                                    |
| tagging evaluations**        |                                                    |
+------------------------------+----------------------------------------------------+
| *sets*                       | Take the unique set of tuples as defined by        |
|                              | **key** across the gold and system data, then      |
|                              | micro-average document-level tp, fp and fn counts. |
+------------------------------+----------------------------------------------------+
| *overlap-{max,sum}{max,sum}* | For tasks in which the gold and system must        |
|                              | produce non-overlapping annotations, these scores  |
|                              | account for partial overlap between gold and       |
|                              | system mentions, as defined for the `LoReHLT`_     |
|                              | evaluation.                                        |
+------------------------------+----------------------------------------------------+
| **Clustering evaluation**    |                                                    |
|                              |                                                    |
+------------------------------+----------------------------------------------------+
| *muc*                        | Count the total number of edits required to        |
|                              | translate from the gold to the system clustering   |
+------------------------------+----------------------------------------------------+
| *b\_cubed*                   | Assess the proportion of each mention's cluster    |
|                              | that is shared between gold and system clusterings |
+------------------------------+----------------------------------------------------+
| *entity\_ceaf*               | Calculate optimal one-to-one alignment between     |
|                              | system and gold clusters based on Dice             |
|                              | coefficient, and get the total aligned score       |
|                              | relative to aligning each cluster with itself      |
+------------------------------+----------------------------------------------------+
| *mention\_ceaf*              | Calculate optimal one-to-one alignment between     |
|                              | system and gold clusters based on number of        |
|                              | overlapping mentions, and get the total aligned    |
|                              | score relative to aligning each cluster with       |
|                              | itself                                             |
+------------------------------+----------------------------------------------------+
| *pairwise*                   | The proportion of true co-clustered mention pairs  |
|                              | that are predicted, etc., as used in computing     |
|                              | BLANC                                              |
+------------------------------+----------------------------------------------------+
| *pairwise\_negative*         | The proportion of true *not* co-clustered mention  |
|                              | pairs that are predicted, etc., as used in         |
|                              | computing BLANC                                    |
+------------------------------+----------------------------------------------------+

.. _LoReHLT: https://www.nist.gov/sites/default/files/documents/itl/iad/mig/LoReHLT16EvalPlan_v1-01.pdf
