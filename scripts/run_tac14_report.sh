#!/usr/bin/env bash
# 
# Prepare score summary in TAC 2014 format
source $(dirname $0)/_common.sh

usage="Usage: $0 OUT_DIR"

if [ "$#" -ne 1 ]; then
    echo $usage
    exit 1
fi

outdir=$1; shift # directory to which results are written

# INITIALISE REPORT HEADER
report=$outdir/00report.tab
echo -e "DiscP\tDiscR\tDiscF\tLinkP\tLinkR\tLinkF\tCEAFmP\tCEAFmR\tCEAFmF\tSystem" \
    > $report

# ADD SYSTEM SCORES
for eval in $outdir/*.evaluation
do
    cat $eval | get_eval_prf strong_typed_mention_match >> $report
    cat $eval | get_eval_prf strong_all_match >> $report
    cat $eval | get_eval_prf mention_ceaf >> $report
    basename $eval \
        | sed 's/\.evaluation//' \
        >> $report
done
