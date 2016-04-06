Entity linking evaluation
=========================

Python evaluation scripts for [TAC](http://www.nist.gov/tac/) entity linking (and related wikification, named entity disambiguation and cross-document coreference tasks).

Requires that `python` (2.7, with Py3k support experimental/partial) be installed on your system with `numpy` (and preferably `scipy` for fast CEAF calculation) and `joblib`. `matplotlib` is required for the `plot-systems` command.

See a list of commands with:
```bash
./nel --help
```

Or install onto your Python path (e.g. with `pip install git+https://github.com/wikilinks/neleval`) then
```bash
python -m neleval --help
```

TAC-KBP 2014 EDL quickstart
===========================

```bash
./scripts/run_tac14_evaluation.sh \
    /path/to/gold.xml \              # TAC14 gold standard queries/mentions
    /path/to/gold.tab \              # TAC14 gold standard link and nil annotations
    /system/output/directory \       # directory containing (only) TAC14 system output files
    /script/output/directory \       # directory to which results are written
    number_of_jobs                   # number of jobs for parallel mode
```

Each file in in the system output directory is scored against gold.tab.

Similar facility is available for TAC-KBP'15 EDL.

More details
============

See [the project wiki](../../wiki) for more details.

References
==========

This project extends the work described in:

* Ben Hachey, Joel Nothman and Will Radford (2014), "(Cheap and easy entity evaluation)[https://aclweb.org/anthology/P/P14/P14-2076]". In Proceedings of ACL.

It was used as the official scorer for Entity (Discovery and) Linking in 2014 and 2015:

* Heng Ji, Joel Nothman and Ben Hachey (2014), "Overview of TAC-KBP2014 Entity Discovery and Linking Tasks", In Proceedings of the Text Analysis Conference.
* Heng Ji, Joel Nothman, Ben Hachey and Radu Florian (2015), "Overview of TAC-KBP2015 Tri-lingual Entity Discovery and Linking Tasks", In Proceedings of the Text Analysis Conference.
