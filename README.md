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
cne SYSTEM evaluate -g GOLD
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

# Systems for comparison

## Robust Disambiguation of Named Entities in Text (Hoffart et al., 2011 - EMNLP)

http://aclweb.org/anthology//D/D11/D11-1072.pdf

Their best system scores 81.82 in `strong_link_match` using gold mentions (they refer to this as micro-averaged precision @1).

## Collective Search for Concept Disambiguation (Pilz & Paass, 2012 - COLING)

http://aclweb.org/anthology//C/C12/C12-1137.pdf

They report an `entity_link_match` F-score of 82.16%, however their paper assumes ``sequential order is taken into account''.

## KORE: Keyphrase Overlap Relatedness for Entity Disambiguation (Hoffart et al., 2012 - CIKM)

http://www.mpi-inf.mpg.de/~sseufert/papers/kore.pdf

They report a `strong_link_match` using gold mentions of 82.31% (using the MW measure).
They ``ignore all mentions that do not have a candidate entity at all'', which we assume to mean excluding NILs, rather than excluding mentions for which search returns no candidates.

## A Framework for Benchmarking Entity-Annotation Systems (Cornolti, 2013 - WWW)

http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40749.pdf

They report a number of figures for TagMe2:
* `weak_link_match` of 58.3% F-score
* `strong_annotation_match` of 56.7% F-score
* `weak_mention_match` of 74.6% F-score
* `entity_link_match` of 65.6% F-score
