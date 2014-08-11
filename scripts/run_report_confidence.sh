#!/usr/bin/env bash
# 
# Prepare summary report per measure with confidence intervals
set -e

usage="Usage: $0 OUT_DIR MEASURE ..."

if [ "$#" -lt 2 ]; then
    echo $usage
    exit 1
fi

outdir=$1; shift # directory to which results are written

for measure in ${@}
do
    echo "INFO preparing $measure report.."

    # INITIALISE REPORT HEADER
    report=$outdir/00report.$measure
    echo -e "99%(\t95%(\t90%(\tscore\t)90%\t)95%\t)99%\tsystem" \
        > $report

    # ADD SYSTEM SCORES
    (
        for sys_eval in $outdir/*.confidence
        do
            cat $sys_eval \
                | grep "^$measure" \
                | grep "fscore" \
                | awk 'BEGIN{OFS="\t"} {print $3,$4,$5,$6,$7,$8,$9}' \
                | tr '\n' '\t'
            basename $sys_eval \
                | sed 's/\.confidence//'
        done
    ) \
        | sort -t$'\t' -k4 -nr \
        >> $report

done
