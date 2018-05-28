
Coreference evaluation
~~~~~~~~~~~~~~~~~~~~~~

Pradhan et al. have published "Scoring Coreference Partitions of
Predicted Mentions: A Reference Implementation" (ACL 2014) describing
their Perl-based `scoring
tool <https://github.com/conll/reference-coreference-scorers>`__ AKA
``scorer.pl``. The neleval package reimplements these measures (MUC,
B-cubed, Entity CEAF, Mention CEAF, and the pairwise coreference and
non-coreference measures that constitute BLANC) with a number of
efficiency improvements, particularly to CEAF, and especially valuable
in the cross-document coreference evaluation setting.

CEAF calculation efficiency
---------------------------

The slow part of calculating CEAF is identifying the maximal linear-sum
assignment between key and response entities, using the Hungarian
Algorithm or a variant thereof. Our implementation is much faster
because: \* scorer.pl manipulates Perl arrays and may be O(n^4), though
I haven't checked, where *n* is the number of key and response entities;
we use an O(n^3) implementation with vectorised NumPy operations in a
very efficient `implementation that was recently adopted into
scipy <http://scipy.github.io/devdocs/generated/scipy.optimize.linear_sum_assignment.html>`__.
Even before further optimisations, this resulted in an order of
magnitude or more runtime improvement over . \* Our *n* is much smaller
in practice. We only perform the Hungarian Algorithm on each strongly
connected component of the assignment graph, and explicitly eliminate
trivial portions of the assignment problem (where there is no confusion
with other entities). So our time complexity is O(n^3) where *n* is the
number of entities in the largest component, rather than the total
number of entities in the evaluation. These optimisations are
particularly valuable in cross-document coref evaluation because the
number of entities is large relative to the number of confusions. \* We
have also made some efficient choices elsewhere in processing, such as
determining entity overlaps using ``scipy.sparse`` matrix
multiplication.

Both our implementation and ``scorer.pl`` support φ3 and φ4 of `Luo's
2005 paper introducing
CEAF <http://www.aclweb.org/anthology/H05-1004>`__. Our mention\_ceaf =
ceafm = φ3. Our entity\_ceaf = ceafe = φ4.

Note on BLANC
-------------

Note that we do not directly report BLANC, although we facilitate
calculation of both its components, using ``pairwise`` and
``pairwise_negative`` aggregates (see our :ref:`command_list_measures` command),
according to Luo et al. 2015's extension of the metric to system
mentions.

Validation of equivalence to reference implementation
-----------------------------------------------------

We have empirically verified the equivalence of metric implementation
between our system and ``scorer.pl``. By pointing the ``COREFSCORER``
environment variable to a local copy of ``scorer.pl``, our system will
`cross-check the results
automatically <https://github.com/wikilinks/neleval/blob/v3.0.0/neleval/coref_metrics.py#L139>`__.
(This will, however, be extremely slow for large CEAF calculations.)

Importing CoNLL 2011-2012 shared task formatted data
----------------------------------------------------

We provide the :ref:`command_prepare_conll_coref` command to import CoNLL
shared task-formatted annotations. We have validated that our metrics match
those produced by Pradhan et al.'s reference implementation for the CoNLL 2011
runs.
