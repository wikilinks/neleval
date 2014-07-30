neleval
=======

Evaluation and error analysis tool for Named Entity Linking / Named Entity Disambiguation / Wikification / Cross-document Coreference Resolution.


This repository is currently being reworked.


Quickstart
==========

Assumes that `python` is installed on your system

```Shell
git clone https://github.com/benhachey/conll03_nel_eval
cd conll03_nel_eval
./cne prepare \
    -k ".*testb.*" \
    -m mappings/map-testb-fromapi-20140227.tsv \
    /path/to/AIDA-YAGO2-dataset.tsv \
    > gold.txt
./cne prepare \
    -k ".*testb.*" \
    -m mappings/map-testb-fromapi-20140227.tsv \
    /path/to/system.txt \
    > system.txt
./cne evaluate \
    -g gold.txt \
    system.txt
```

See [project wiki](https://github.com/benhachey/conll03_nel_eval/wiki/) for details.
