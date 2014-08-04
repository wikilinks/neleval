Entity linking evaluation
=========================

Python evaluation scripts for [TAC](http://www.nist.gov/tac/) and [CoNLL-YAGO](http://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/) entity linking data, and related wikification, named entity disambiguation and cross-document coreference tasks.

TAC13 quickstart
================

Assumes that `python` is installed on your system with `numpy` (and preferably `scipy` for fast CEAF calculation) and `joblib`.

```Shell
git clone https://github.com/wikilinks/neleval
cd neleval
./scripts/run_tac13_evaluation.sh \
    path/to/queries.xml \
    path/to/gold.tab \
    path/to/sys/out/dir \
    path/to/eval/out/dir \
    NUM_JOBS_FOR_PARALLEL_MODE
```

Each file in in the system output directory is scored against gold.tab.
