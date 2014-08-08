#!/usr/bin/env bash
# 
# Run TAC 2013 evaluation
set -e

usage="Usage: $0 GOLD_XML GOLD_TAB SYSTEMS_DIR SCORES_DIR OUT_DIR"

if [ "$#" -ne 5 ]; then
    echo $usage
    exit 1
fi

goldx=$1; shift # gold standard queries/mentions (XML)
goldt=$1; shift # gold standard link annotations (tab-separated)
sysdir=$1; shift  # directory containing output from systems
scodir=$1; shift # directory containing official scores
outdir=$1; shift # directory to which results are written

SCR=`dirname $0`

SCOFN="tac_2013_kbp_english_entity_linking_evaluation_KB_links.tab.scores"
JOBS=8 # number of jobs for parallel mode


# CALCULATE 2013 SCORES
$SCR/run_tac13_evaluation.sh $goldx $goldt $sysdir $outdir $JOBS
report=$outdir/00report.tab
if [ ! -e $report ]
then
    echo "ERROR $report does not exist"
    exit
fi


# PREPARE OFFICIAL 2013 SCORES
scores=$scodir/$SCOFN
official=$outdir/00official.tab
cat $scores \
    | egrep -v '^[0-9]* queries' \
    | head -1 \
    > $official
cat $scores \
    | egrep -v '^[0-9]* queries' \
    | tail -n +2 \
    | sort \
    >> $official


# COMPARE CALCULATED TO OFFICIAL
echo "TEST compare to official 2013 results.."
if [ "" != "`diff $official $report`" ]
then
    difff=$outdir/00diff.txt
    diff -y $official $report \
        > $difff
    echo "FAIL see $difff"
else
    echo "PASS"
fi


echo "..done."

