Entity linking evaluation
=========================

Python evaluation scripts for [TAC](http://www.nist.gov/tac/) and [CoNLL-YAGO](http://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/) entity linking data, and related wikification, named entity disambiguation and cross-document coreference tasks.

TAC13 quickstart
================

Assumes that `python` (2.7, with Py3k support soon) is installed on your system with `numpy` (and preferably `scipy` for fast CEAF calculation) and `joblib`.

```bash
./scripts/run_tac14_evaluation.sh \
    /path/to/gold.xml \              # TAC14 gold standard queries/mentions
    /path/to/gold.tab \              # TAC14 gold standard link and nil annotations
    /system/output/directory \       # directory containing (only) TAC14 system output files
    /script/output/directory \       # directory to which results are written
    number_of_jobs                   # number of jobs for parallel mode
```

Each file in in the system output directory is scored against gold.tab.

See [the project wiki](../../wiki) for more details.
