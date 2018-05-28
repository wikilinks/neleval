.. _formats:

File formats
~~~~~~~~~~~~

``neleval`` annotations format
==============================

Annotations provided as input to most ``neleval`` tools (e.g.
:ref:`command_evaluate`) consists of a tab-delimited file.
Each line corresponds to an entity mention, and has the following
columns:

document ID : str
    Should not contain whitespace.

mention start offset : int
    The units are arbitrary unless overlap aggregators are used (see
    :ref:`measure_aggregator`).

mention end offset : int
    This should be inclusive of the last unit.
    Thus if offsets are character counts, a mention with text "Go" may have
    start offset 3 and end offset 4 (unlike Python slice notation).

entity ID : str
    Should not contain whitespace.
    Should start with ``NIL`` for an arbitrary (cluster) identifier, or another
    string for a KB identifier.

score : float
    ..

type : str
    An entity type label

If there is more than one candidate, more (entity ID, score, type) column
triples may be added, separated by tabs.

TAC data
========

The TAC entity linking data is available to participants in the `entity
linking track <http://nlp.cs.rpi.edu/kbp/2014/>`__ of `NIST's knowledge
base population shared task <http://tac.nist.gov/2014/KBP/>`__. The
data format is described briefly below. For more details, see `the
entity linking task
definition <http://nlp.cs.rpi.edu/kbp/2014/task.html>`__.

.. _format_tac14:

TAC 2014
........

In 2014, systems must provide two files: (1) an ``xml`` file containing
entity mentions and (2) a ``tab`` file containing linking and nil
clustering output.

Mention query XML
-----------------

The mention ``xml`` file includes a query element for each mention. This
element must have an ``id`` attribute with a unique value as well as
``docid`` (document identifier), ``beg`` (start offset), ``end`` (end
offset) elements:

.. code:: xml

    <kbpentlink>
        <query id="EDL14_ENG_TRAINING_0001"> 
            <name>Xenophon</name> 
            <docid>bolt-eng-DF-170-181122-8792777</docid> 
            <beg>22103</beg> 
            <end>22110</end> 
        </query>
        <query id="EDL14_ENG_TRAINING_0002">
            <name>Richmond</name>
            <docid>APW_ENG_20090826.0903</docid>
            <beg>340</beg>
            <end>347</end>
        </query>
        ...
    </kbpentlink>

Note that offsets should be character offsets over the utf8-encoded sgml
source files. The end offset should be the last character that is
included in the span.

Link ID file
------------

The tab-separated link ID file includes a line for each mention. Each
line includes several fields: ``query_id`` (matching the ``id``
attribute on a ``query`` element in the corresponding mentions ``xml``
file), ``kb_or_nil_id`` (a knowledge base or nil cluster identifier),
``entity_type`` (the type is required for 2014 link evaluation), and
``score`` (a confidence value, optional):

::

    EDL14_ENG_TRAINING_0001    NIL0001     PER    1.0
    EDL14_ENG_TRAINING_0002    E0604067    GPE    1.0

Note that it is possible to provide more than one response for a given
mention by adding extra lines. However, the current set of evaluation
measures only consider one response per mention (the one with the
highest score).

TAC 2009-2013
.............

Before 2014, the mention ``xml`` was provided and systems only need to
output a tab-separated link ID file containing ``query_id``,
``kb_or_nil_id``, and ``score`` fields. To evaluate on these data sets,
first add a ``ne_type`` field as per the 2014 format. Then use the gold
``xml`` file when converting system output to evaluation
format with :ref:`command_prepare_tac`.

Note that when using 2011 data, the end offset is the first character
that is not part of the span (rather than the last character that is
included in the span).
