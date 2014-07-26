#!/usr/bin/env bash

usage="Usage: $0 GOLD_XML GOLD_TAB SYS_XML SYS_TAB"

if [ "$#" -ne 4 ]; then
    echo $usage
    exit 1
fi

goldx=$1; shift # gold standard queries/mentions (XML)
goldt=$1; shift # gold standard link annotations (tab-separated)
sysx=$1; shift  # system queries/mentions (XML)
syst=$1; shift  # system link annotations (tab-separated)

outdir=tacout
mkdir tacout # output directory


# STEP 1: CONVERT TO EVALUATION FORMAT
gold=$outdir/gold.combined.tsv
./cne prepare-tac -q $goldx $goldt > $gold
sys=$outdir/sys.combined.tsv
./cne prepare-tac -q $sysx $syst > $sys


# STEP 2: EVALUATE
eval=$outdir/sys.eval
./cne evaluate -l tac -c tac -f 'json_format' -g $gold $sys #> $eval


# STEP 3: TEST SIGNIFICANCE (TODO)


# STEP 4: EVALUATE BY ENTITY TYPE (TODO)


# STEP 5: EVALUATE BY DOCUMENT GENRE (TODO)


# STEP 5 RUN ERROR TYPE ANALYSIS (TODO)


