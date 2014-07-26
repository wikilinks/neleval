conll03_nel_eval
================

Python evaluation scripts for [AIDA-formatted CoNLL NER data](https://github.com/wikilinks/conll03_nel_eval/wiki/Data%20set)

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
