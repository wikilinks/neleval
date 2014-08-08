#!/usr/bin/env bash
# 
# Prepare score summary in TAC 2014 format
set -e

usage="Usage: $0 OUT_DIR"

if [ "$#" -ne 1 ]; then
    echo $usage
    exit 1
fi

outdir=$1; shift # directory to which results are written

# INITIALISE REPORT HEADER
report=$outdir/00report.tab
echo -e "WikiF1\tCEAFmP\tCEAFmR\tCEAFmF1\tSystem" \
    > $report

# ADD SYSTEM SCORES
for eval in $outdir/*.evaluation
do
    cat $eval \
        | grep -P '\tstrong_typed_all_match$' \
        | cut -f7 \
        | tr '\n' '\t' \
        >> $report
    cat $eval \
        | grep -P '\tmention_ceaf$' \
        | cut -f5,6,7 \
        | tr '\n' '\t' \
        >> $report
    basename $eval \
        | sed 's/\.evaluation//' \
        >> $report
done
