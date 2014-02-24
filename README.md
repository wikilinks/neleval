conll03_nel_eval
================

Python evaluation scripts for AIDA-formatted CoNLL data

Simple installation
===================

* Assumes that `python` is installed on your system
* `SYSTEM` and `GOLD` are your system output and gold-standard in CoNLL/AIDA format

```Shell
git clone https://github.com/benhachey/conll03_nel_eval
cd conll03_nel_eval
cne evaluate -g GOLD SYSTEM
```

Installing as a module
======================

Pip should be able to install directly from this repository:
```Shell
mkdir some_project
cd some_project
virtualenv ve
source ve/bin/activate
pip install git+git://github.com/benhachey/conll03_nel_eval.git#egg=CNE
```

Details
=======

See [project wiki](https://github.com/benhachey/conll03_nel_eval/wiki/) for details.



## Filtering datasets

The distributed gold-standard includes three splits: train, testa and testb. To filter out some of these splits, run:
```Shell
cne filter -s testb gold.txt > gold.testb.txt
```

Wikipedia (and other KBs) change over time, including page titles. A system using a more recent version of Wikipedia may lose points for using a newer title. Luckily, Wikipedia redirects can often be used to map between titles in different versions.

The map script can be used to map link titles in SYSTEM and GOLD to a common version:

```Shell
cne filter -m MAP SYSTEM > SYSTEM.mapped
```

The `MAP` file should contain lines corresponding to titles from the newer version. The first column contains the newer title and any following tab-separated columns contain names that should map to the newer title (e.g., titles of redirect pages that point to the newer title).

The fetch_map script can be used to generate a current redirect mapping using the Wikipedia API:

```Shell
cne GOLD fetch_map > MAP
```
