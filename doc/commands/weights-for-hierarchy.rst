.. _command_weights_for_hierarchy:

``neleval weights-for-hierarchy``
---------------------------------

Translate a hierarchy of types into a sparse matrix of type-pair weights

See :ref:`approx_type`.

Usage summary
.............

.. command-output:: neleval weights-for-hierarchy --help

Converting JSON type hierarchy to weights
.........................................

.. command-output:: bash -c "\
        neleval weights-for-hierarchy --decay 0.5 <( \
        echo '{\"root\": [\"A\", \"B\"], \"A\": [\"A1\", \"A2\"], \"B\": [\"B1\"], \"B1\": [\"B1i\"]}' \
        ) \
        "

These weights can be applied to evaluation with :ref:`command_evaluate`'s
``--type-weight`` option.
