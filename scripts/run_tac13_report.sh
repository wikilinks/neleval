#!/usr/bin/env bash
# 
# Prepare score summary in TAC 2013 format
set -e

usage="Usage: $0 OUT_DIR"

if [ "$#" -ne 1 ]; then
    echo $usage
    exit 1
fi

outdir=$1; shift # directory to which results are written

# INITIALISE REPORT HEADER
report=$outdir/00report.tab
echo -e "system\tKBP2010 micro-average\tB^3 Precision\tB^3 Recall\tB^3 F1" \
    > $report

# ADD SYSTEM SCORES
# TODO add B^3+
for eval in $outdir/*.evaluation
do
    basename $eval \
        | sed 's/\.evaluation//' \
        | tr '\n' '\t' \
        >> $report
    cat $eval \
        | egrep '(strong_all_match|b_cubed)' \
        | cut -f5,6,7,8 \
        | tr '\n' '\t' \
        | cut -f2,5,6,7 \
        >> $report
done
