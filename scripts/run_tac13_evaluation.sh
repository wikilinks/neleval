#!/usr/bin/env bash
# 
# Run TAC 2013 evaluation

usage="Usage: $0 GOLD_XML GOLD_TAB SYSTEMS_DIR OUT_DIR NUM_JOBS"

if [ "$#" -ne 5 ]; then
    echo $usage
    exit 1
fi

goldx=$1; shift # gold standard queries/mentions (XML)
goldt=$1; shift # gold standard link annotations (tab-separated)
sysdir=$1; shift # directory containing output from systems
outdir=$1; shift # directory to which results are written
jobs=$1; shift # number of jobs for parallel mode

SCR=`dirname $0`


# CONVERT GOLD TO EVALUATION FORMAT
echo "INFO Converting gold to evaluation format.."
gtab=$outdir/gold.tab
cat $goldt \
    | cut -f1,2,3 \
    > $gtab
#rm $gtab
gold=$outdir/gold.combined.tsv
./nel prepare-tac \
    -q $goldx \
    $gtab \
    > $gold


# CONVERT SYSTEMS TO EVALUATION FORMAT
echo "INFO Converting systems to evaluation format.."
ls $sysdir/* \
    | xargs -n 1 -P $jobs $SCR/run_tac13_prepare.sh $goldx $outdir


# TODO filter (e.g., to evaluate on PER or news only)!!!


# EVALUATE
echo "INFO Evaluating systems.."
ls $outdir/*.combined.tsv \
    | grep -v "gold\.combined\.tsv$" \
    | xargs -n 1 -P $jobs $SCR/run_evaluate.sh $gold


# PREPARE REPORT CSV FILES
# TODO add B^3+
echo "INFO Preparing summary report.."
report=$outdir/00report.tab
echo -e "system\tKBP2010 micro-average\tB^3 Precision\tB^3 Recall\tB^3 F1" \
    > $report
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
