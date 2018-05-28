Basic measures
~~~~~~~~~~~~~~

The evaluation tool provides a range of linking and clustering
evaluation measures. These are described briefly below and listed by the
``nel list-measures`` command. For more details of correspondences
between linking measures here and in the literature, see `Hachey et al.
(2014) <http://www.aclweb.org/anthology/P14-2076.pdf>`__. For
clustering, see `Pradhan et al.
(2014) <http://www.aclweb.org/anthology/P14-2006.pdf>`__. For a quick
reference, see our `cheatsheet <Cheatsheet>`__. (As described there,
evaluation can be performed across the whole corpus, or with separate
scores for each document/type as well as micro- and macro-averages
across all types/docs.)

Official TAC 2014 measures
==========================

TAC 2014 reports two official measures, one for linking/wikification
and one for nil clustering. For more detail, see `the TAC 2014 scoring
page <http://nlp.cs.rpi.edu/kbp/2014/scoring.html>`__.

Linking evaluation
------------------

``strong_typed_all_match`` is a micro-averaged evaluation of all
mentions. A mention is counted as correct if it is a correct link or a
correct nil. A correct link must have the same span, entity type, and KB
identifier as a gold link. A correct nil must have the same span as a
gold nil. This is the official linking evaluation measure for TAC 2014.

Clustering evaluation
---------------------

``mention_ceaf`` is based on a one-to-one alignment between system and
gold clusters — both KB and nil. It computes an optimal mapping based on
overlap between system-gold cluster pairs. System and gold mentions must
have the same span to affect the alignment. Unmatched mentions also
affect precision and recall.

Additional diagnostic measures
==============================

The evaluation tool also provides a number of diagnostic measures
available to isolate performance of system components and compare to
numbers reported elsewhere in the literature.

Mention detection evaluation
----------------------------

``strong_mention_match`` is a micro-averaged evaluation of entity
mentions. A system span must match a gold span exactly to be counted as
correct.

``strong_typed_mention_match`` additionally requires the correct entity
type. This is equivalent to the CoNLL NER evaluation (`Tjong Kim Sang &
De Meulder,
2003 <https://www.clips.uantwerpen.be/conll2003/pdf/14247tjo.pdf>`__).

``strong_linked_mention_match`` is the same as ``strong_mention_match``
but only considers non-nil mentions that are linked to KB identifier.

Measures sensitive to partial overlap between the system and gold
mentions, using the `LoReHLT
metric <https://www.nist.gov/sites/default/files/documents/itl/iad/mig/LoReHLT16EvalPlan_v1-01.pdf>`__
can be constructed with aggregates such as ``overlap-sumsum``. See the
:ref:`detailed_measures`.

Linking evaluation
------------------

``strong_link_match`` is a micro-averaged evaluation of links. A system
link must have the same span and KB identifier as a gold link to be
counted as correct. This is equivalent to `Cornolti et al.'s
(2013) <http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40749.pdf>`__
strong annotation match. Recall here is equivalent to KB accuracy from
TAC tasks before 2014.

``strong_nil_match`` is a micro-averaged evaluation of nil mentions. A
system nil must have the same span as a gold nil to be counted as
correct. Recall here is equivalent to nil accuracy from TAC tasks before
2014.

``strong_all_match`` is a micro-averaged link evaluation of all
mentions. A mention is counted as correct if is either a link match or a
nil match as defined above. This is equivalent to overall accuracy from
TAC tasks before 2014.

Document-level tagging evaluation
---------------------------------

``entity_match`` is a micro-averaged document-level set-of-titles
measure. It is the same as entity match reported by `Cornolti et al.
(2013) <http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40749.pdf>`__.

Clustering evaluation
---------------------

``entity_ceaf`` — like ``mention_ceaf`` — is based on a one-to-one
alignment between system and gold entity clusters. Here system-gold
cluster pairs are scored by their Dice coefficient.

``b_cubed`` assesses the proportion of each mention's cluster that is
shared between gold and predicted clusterings.

``b_cubed_plus`` is identical to ``b_cubed``, but additionally requires
a correct KB identifier for non-nil mentions.

``muc`` counts the number of edits required to translate the gold
clustering into the prediction.

``pairwise`` measures the proportion of mention pairs occurring in the
same cluster in both gold and predicted clusterings. It is similar to
the Rand Index.

For more detail, see `Pradhan et al.'s
(2014) <http://www.aclweb.org/anthology/P14-2006.pdf>`__ excellent
overview of clustering measures for coreference evaluation, and our
`Coreference\_Evaluation <implementation%20notes>`__.

Custom measures
---------------

Our scorer supports specification of some custom evaluation measures.
See :ref:`command_list_measures`.

References
==========

Cornolti et al. (2013). `A framework for benchmarking entity-annotation
systems <http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40749.pdf>`__.
In WWW.

Hachey et al. (2014). `Cheap and easy entity
evaluation <http://www.aclweb.org/anthology/P14-2076.pdf>`__. In ACL.

Ji & Grishman (2011). `Knowledge base population: successful approaches
and challenges <http://www.aclweb.org/anthology/P11-1115.pdf>`__. In
ACL.

Pradhan et al. (2014). `Scoring Coreference Partitions of Predicted
Mentions: A Reference
Implementation <http://www.aclweb.org/anthology/P14-2006.pdf>`__. In
ACL.

Tjong Kim Sang & De Meulder (2003). `Introduction to the CoNLL-2003
shared task: Language-independent named entity
recognition <https://www.clips.uantwerpen.be/conll2003/pdf/14247tjo.pdf>`__. In
CoNLL.
