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

