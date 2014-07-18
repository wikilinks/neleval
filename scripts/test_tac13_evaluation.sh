#!/usr/bin/env bash
# 
# Run TAC 2013 evaluation

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

SCOFN="tac_2013_kbp_english_entity_linking_evaluation_KB_links.tab.scores"


# CALCULATE 2013 SCORES
scr=`dirname $0`
$scr/run_tac13_evaluation.sh $goldx $goldt $sysdir $outdir
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
    | cut -f1,2,3,4,5 \
    > $official
cat $scores \
    | egrep -v '^[0-9]* queries' \
    | awk '{if (NR>1) print}' \
    | cut -f1,2,3,4,5 \
    | sort \
    >> $official


# COMPARE CALCULATED TO OFFICIAL
echo "TEST compare to official 2013 results.."
if [ `diff $official $report` ]
then
    difff=$outdir/00diff.txt
    diff -y $official $report \
	> $difff
    echo "FAIL see $difff"
else
    echo "PASS"
fi


echo "..done."