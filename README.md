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

# Quick reference

## Evaluate

The evaluate script reads `SYSTEM` output in AIDA/CoNLL format and calculates a number of evaluation measures:

```Shell
cne evaluate -g GOLD SYSTEM
```

`link_entity_match` is a micro-averaged document-level set-of-titles measure. It is the same as entity match reported Cornolti et al. (2013). TODO Same as Ratinov???

`weak_link_match` is a micro-averaged evaluation of links. The system mention extent must have at least one token in common with the aligned gold mention and the link must be to the same KB title. It is the same as weak annotation match reported in Cornolti et al. (2013).

`strong_link_match` is the same as `weak_link_match` but is stricter, requiring the system and gold mentions to be exactly the same. It is the same as strong annotation match reported in Cornolti et al. (2013).

`weak_mention_match` is a micro-averaged evaluation of entity mentions. The system mention extent must have at least one token in common with the aligned gold mention. It is the same as weak mention match reported in Cornolti et al. (2013).

`strong_mention_match` is the same as `weak_mention_match` but is stricter, requiring the system and gold mentions to be exactly the same.

`weak_nil_match` is a micro-averaged evaluation of unlinked entity mentions. The system mention extent must have at least one token in common with the aligned gold mention and must be unlinked. This is useful for systems that perform NER and NIL handling in addition to KB linking.

`strong_nil_match` is the same as `weak_nil_match` but is stricter, requiring the system and gold mentions to be exactly the same.

`weak_all_match` is a convenience metric that combines `weak_link_match` and `weak_nil_match` into a single micro-averaged score.

`strong_all_match` is a convenience metric that combines `strong_link_match` and `strong_nil_match` into a single micro-averaged score.

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

# Systems for comparison

## Robust Disambiguation of Named Entities in Text (Hoffart et al., 2011 - EMNLP)
  
http://aclweb.org/anthology//D/D11/D11-1072.pdf

Their best system scores 81.82% precision in `strong_link_match` using gold mentions (they refer to this as micro-averaged precision @1).

## Collective Search for Concept Disambiguation (Pilz & Paass, 2012 - COLING)

http://aclweb.org/anthology//C/C12/C12-1137.pdf

They report an `entity_link_match` F-score of 82.16%, however their paper assumes ``sequential order is taken into account''.

## KORE: Keyphrase Overlap Relatedness for Entity Disambiguation (Hoffart et al., 2012 - CIKM)

http://www.mpi-inf.mpg.de/~sseufert/papers/kore.pdf

They report a `strong_link_match` using gold mentions of 82.31% precision (using the MW measure).
They ``ignore all mentions that do not have a candidate entity at all'', which we assume to mean excluding NILs, rather than excluding mentions for which search returns no candidates.

## A Framework for Benchmarking Entity-Annotation Systems (Cornolti, 2013 - WWW)

http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40749.pdf

They report a number of figures for TagMe2:
* `weak_link_match` of 58.3% F-score
* `strong_annotation_match` of 56.7% F-score
* `weak_mention_match` of 74.6% F-score
* `entity_link_match` of 65.6% F-score

# System output format

To evaluate your system output against the gold-standard, you will need to output in tab-separated format.
* Each document is started with the `-DOCSTART- (some_doc_id)` line, where `some_doc_id` might be something like `1163testb SOCCER`
* Each sentence is separated by a blank line
* Each document is separated by a blank line
* Each token is on its own line (we re-use the gold-standard tokenisation)

The column ordering for token lines is:
* Token
* Mention span: we use `IOB`
* Mention text: this is a bit redundant, but a sanity check when reading the output (`text == ' '.join(mentiontokens`)
* Entity identifier: where a mention is linked to the KB, this will be the id/title (e.g., a Wikipedia title). Where the mention is a `NIL`, this column should be blank

```
-DOCSTART- (some_doc_id)
Some
headline
about
two
Named	B	Named Entities	Named_Entity
Entities	I	Named Entities	Named_Entity
.

By
John	B	John Smith
Smith	I	John Smith
```
