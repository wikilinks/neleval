#!/usr/bin/env bash
# 
# Script to p TAC 2013 evaluation

usage="Usage: $0 GOLD_XML GOLD_TAB SYS_DIR"

if [ "$#" -ne 3 ]; then
    echo $usage
    exit 1
fi

goldx=$1; shift # gold standard queries/mentions (XML)
goldt=$1; shift # gold standard link annotations (tab-separated)
sysd=$1; shift  # directory structure containing output from systems

outdir=tacout
mkdir tacout # output directory


# CONVERT GOLD TO EVALUATION FORMAT
echo "Converting gold to evaluation format.."
gtab=$outdir/gold.tab
cat $goldt \
    | cut -f1,2,3 \
    > $gtab
#rm $gtab
gold=$outdir/gold.combined.tsv
./cne prepare-tac -q $goldx $gtab > $gold


# EVAL EACH RUN IN $sysd
for runt in $sysd/*
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

    # TEST SIGNIFICANCE
    # TODO

    # EVALUATE BY ENTITY TYPE
    # TODO

    # EVALUATE BY DOCUMENT GENRE
    # TODO

    # RUN ERROR ANALYSIS
    # TODO

done


# PREPARE REPORT CSV FILES
report=$outdir/00report.tab
echo -e "system\tKB2010 micro-average\tB^3 Precision\tB^3 Recall\tB^3 F1" \
    > $report
for eval in `ls $outdir/*.eval | sort -f`
do
    (
	basename $eval \
	    | sed 's/\.eval//' \
	    | tr '\n' '\t'
	cat $eval \
	    | egrep '(strong_all_match|b_cubed)' \
	    | cut -f5,6,7,8 \
	    | tr '\n' '\t' \
	    | cut -f2,5,6,7
    ) >> $report
done


echo "..done."