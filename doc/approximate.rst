
Approximate matching
~~~~~~~~~~~~~~~~~~~~

:ref:`Measures <detailed_measures>` ordinarily score 1 when gold and system
annotations exist that have an exact match for all elements of the
:ref:`key <measure_key>`.

For some kinds of measure it is possible to award partial matches for:

* mention pairs with overlapping, but not identical, spans
* mention pairs with related, but not identical, entity types
* mention pairs with related, but not identical, KB entries (disambiguands)

Overlapping spans
-----------------

To give partial award to overlapping gold and system mentions, we use the
scheme developed by Ryan Gabbard of BBN for `LoReHLT`_:

    We award systems for partial matches according to the degree of 
    character overlap between system and key names. The partial match scoring algorithm has two 
    parameters: the recall overlap strategy and the precision overlap strategy.

    * The per-name recall score of a name in the answer key is the fraction of
      its characters which overlap with the system name set according to the
      recall overlap strategy parameter. For the "MAX" strategy, this will be
      the characters overlapping with the single system name with maximum
      overlap. For the "SUM" strategy, this will be the number of its
      characters which overlap with any system mention.
    * The recall score for a system is the mean of the per-name recall scores
      for all names in the answer key.
    * The per-name precision score of a name in the answer key is the fraction
      of its characters overlapped by the reference set, where ”overlapping" is
      determined by the precision overlap strategy in the same manner as above
      for recall.
    * The precision score for a system is the mean of the per-name precision scores for all names in 
      the answer key.

This applies to measures with :ref:`aggregator <measure_aggregator>`:

* ``overlap-maxmax`` for recall and precision overlap strategies both MAX
* ``overlap-maxsum`` for recall overlap strategy MAX and precision overlap strategy SUM
* ``overlap-summax`` for recall overlap strategy SUM and precision overlap strategy MAx
* ``overlap-sumsum`` for recall and precision overlap strategies both SUM

In the following example, the gold standard includes a mention from character 1 to 10 and another from 12 to 12. The system includes a mention from 1 to 5 and another from 6 to 12.

.. command-output:: bash -c "\
    neleval evaluate \
    -m overlap-maxmax::span \
    -m overlap-maxsum::span \
    -m overlap-summax::span \
    -m overlap-sumsum::span \
    -m sets::span \
    -g <(echo -e 'd\t1\t10\nd\t12\t12') \
       <(echo -e 'd\t1\t5\nd\t6\t12')"
    :nostderr:

TODO: flesh out calculation

Caveats:

* All mentions within the gold annotation must be non-overlapping.
* All mentions within the system annotation must be non-overlapping.
* There is (currently) no equivalent implementation for clustering metrics.

.. _LoReHLT: https://www.nist.gov/sites/default/files/documents/itl/iad/mig/LoReHLT16EvalPlan_v1-01.pdf
