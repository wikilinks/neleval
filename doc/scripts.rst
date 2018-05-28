
.. _scripts:

Convenience scripts for TAC KBP evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository includes `a number of convenience
scripts <https://github.com/wikilinks/neleval/tree/master/scripts>`__ to illustrate and automate common
usage.

Basic evaluation and reporting
==============================

The basic evaluation scripts automate the following workflow:

1. `convert the gold data to the evaluation tool
   format <Usage#convert-gold-standard-to-evaluation-format>`__,
2. `convert each system run output to the evaluation tool
   format <Usage#convert-system-output-to-evaluation-format>`__,
3. `evaluate each system run <Usage#evaluate-system-output>`__.

The following are written to the output directory:

-  detailed evaluation report for each run (\*.evaluation),
-  summary evaluation report for comparing runs (00report.tab).

Usage for TAC14 output format:

.. code:: bash

    ./scripts/run_tac14_evaluation.sh \
        /path/to/gold.xml \              # TAC14 gold standard queries/mentions
        /path/to/gold.tab \              # TAC14 gold standard link and nil annotations
        /system/output/directory \       # directory containing (only) TAC14 system output files
        /script/output/directory \       # directory to which results are written
        number_of_jobs                   # number of jobs for parallel mode

Usage for TAC13 output format:

.. code:: bash

    ./scripts/run_tac13_evaluation.sh \
        /path/to/gold.xml \              # TAC13 gold standard queries/mentions
        /path/to/gold.tab \              # TAC13 gold standard link and nil annotations
        /system/output/directory \       # directory containing (only) TAC13 system output files
        /script/output/directory \       # directory to which results are written
        number_of_jobs                   # number of jobs for parallel mode 

Analysis and confidence reporting
=================================

The analysis scripts automate the following workflow:

1. `run the basic evaluation <#basic-evaluation-scripts>`__,
2. `calculate confidence intervals for each system
   run <Usage#calculate-confidence-intervals>`__,
3. `count errors for each system run (nil-as-link, link-as-nil,
   wrong-link counts) <Usage#analyze-error-types>`__.

The following are written to the output directory:

-  detailed evaluation report for each run (\*.evaluation),
-  summary evaluation report for comparing runs (00report.tab),
-  detailed confidence interval report for each run (\*.confidence),
-  summary confidence interval report for comparing runs (00report.\*),
-  error type distribution for each run (\*.analysis).

Usage for TAC14 output format:

.. code:: bash

    ./scripts/run_tac14_all.sh \
        /path/to/gold.xml \              # TAC14 gold standard queries/mentions
        /path/to/gold.tab \              # TAC14 gold standard link and nil annotations
        /system/output/directory \       # directory containing (only) TAC14 system output files
        /script/output/directory         # directory to which results are written

Usage for TAC13 output format:

.. code:: bash

        /path/to/gold.xml \              # TAC13 gold standard queries/mentions
        /path/to/gold.tab \              # TAC13 gold standard link and nil annotations
        /system/output/directory \       # directory containing (only) TAC13 system output files
        /script/output/directory         # directory to which results are written

Filtered evaluation
===================

The filtered evaluation scripts automate the following workflow:

1. filter gold data to include a specific subset of instances,
2. filter each system run to include a specific subset of instances,
3. `run the basic evaluation over subset
   data <#basic-evaluation-and-reporting>`__.

The following are written to an output directory for each subset:

-  detailed evaluation report for each run (\*.evaluation),
-  summary evaluation report for comparing runs (00report.tab).

The following subsets/directorys are defined:

-  PER - mentions with person entity type,
-  ORG - mentions with organisation entity type,
-  GPE - mentions with geo-political entity type,
-  NW - mentions from newswire documents,
-  WB - mentions from newsgroup and blog documents,
-  DF - mentions from discussion forum documents,
-  entity-document type combinations (PER\_NW, PER\_WB, PER\_DF,
   ORG\_NW, etc.).

Usage for TAC14 output format:

.. code:: bash

    ./scripts/run_tac14_filtered.sh \
        /path/to/gold.xml \              # TAC14 gold standard queries/mentions
        /path/to/gold.tab \              # TAC14 gold standard link and nil annotations
        /system/output/directory \       # directory containing (only) TAC14 system output files
        /script/output/directory         # directory to which results are written

Usage for TAC13 output format:

.. code:: bash

    ./scripts/run_tac13_filtered.sh \
        /path/to/gold.xml \              # TAC13 gold standard queries/mentions
        /path/to/gold.tab \              # TAC13 gold standard link and nil annotations
        /system/output/directory \       # directory containing (only) TAC13 system output files
        /script/output/directory         # directory to which results are written

Test evaluation on TAC 2013 data
================================

The test evaluation script automates the following workflow:

1. `run the basic evaluation <#basic-evaluation-scripts>`__,
2. compare evaluation output to official TAC13 results.

The following are written to the output directory:

-  detailed evaluation report for each run (\*.evaluation),
-  summary evaluation report for comparing runs (00report.tab),
-  copy of the official results sorted for comparison (00official.tab),
-  a diff report if the test fails (00diff.txt).

Usage for TAC13 official results:

.. code:: bash

    ./scripts/test_tac13_evaluation.sh \
        /path/to/gold.xml \              # TAC13 gold standard queries/mentions
        /path/to/gold.tab \              # TAC13 gold standard link and nil annotations
        /system/output/directory \       # directory containing (only) TAC13 system output files
        /system/scores/directory \       # directory containing official score summary reports
        /script/output/directory         # directory to which results are written

The gold data from TAC13 is distributed by LDC. When running the test
evaluation script, provide: \*
``LDC2013E90_TAC_2013_KBP_English_Entity_Linking_Evaluation_Queries_and_Knowledge_Base_Links_V1.1/data/tac_2013_kbp_english_entity_linking_evaluation_queries.xml``,
\*
``LDC2013E90_TAC_2013_KBP_English_Entity_Linking_Evaluation_Queries_and_Knowledge_Base_Links_V1.1/data/tac_2013_kbp_english_entity_linking_evaluation_KB_links.tab``.

The system data from TAC13 is distributed by NIST. When running the test
evaluation script, provide: \*
``KBP2013_English_Entity_Linking_Evaluation_Results/KBP2013_english_entity-linking_runs``,
\*
``KBP2013_English_Entity_Linking_Evaluation_Results/KBP2013_english_entity-linking_scores``.
