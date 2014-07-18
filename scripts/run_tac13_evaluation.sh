#!/usr/bin/env bash
# 
# Run TAC 2013 evaluation

usage="Usage: $0 GOLD_XML GOLD_TAB SYSTEMS_DIR OUT_DIR"

if [ "$#" -ne 4 ]; then
    echo $usage
    exit 1
fi

goldx=$1; shift # gold standard queries/mentions (XML)
goldt=$1; shift # gold standard link annotations (tab-separated)
sysdir=$1; shift  # directory containing output from systems
outdir=$1; shift # directory to which results are written


# CONVERT GOLD TO EVALUATION FORMAT
echo "Converting gold to evaluation format.."
gtab=$outdir/gold.tab
cat $goldt \
    | cut -f1,2,3 \
    > $gtab
#rm $gtab
gold=$outdir/gold.combined.tsv
./cne prepare-tac -q $goldx $gtab > $gold


# EVAL EACH RUN IN $sysdir
for runt in $sysdir/*
do
    run=`basename $runt`
    echo "Evaluating run $run.."

    # CONVERT TO EVALUATION FORMAT
    stab=$run.tab
    cat $runt \
	| awk 'BEGIN{OFS="\t"} {print $1,$2,"NA",$3}' \
	> $stab
    #rm $stab
    sys=$outdir/$run.combined.tsv
    ./cne prepare-tac -q $goldx $stab > $sys

    # EVALUATE
    eval=$outdir/$run.eval
    ./cne evaluate -l tac -c tac -f 'tab_format' -g $gold $sys > $eval

done


# PREPARE REPORT CSV FILES
echo "Preparing summary report.."
report=$outdir/00report.tab
echo -e "system\tKBP2010 micro-average\tB^3 Precision\tB^3 Recall\tB^3 F1" \
    > $report
for eval in $outdir/*.eval
do
    basename $eval \
	| sed 's/\.eval//' \
	| tr '\n' '\t' \
	>> $report
    cat $eval \
	| egrep '(strong_all_match|b_cubed)' \
	| cut -f5,6,7,8 \
	| tr '\n' '\t' \
	| cut -f2,5,6,7 \
	>> $report
done


echo "..done."