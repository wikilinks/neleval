#!/usr/bin/env bash
set -e

usage="Usage: $0 GOLD_XML GOLD_TAB SYSTEMS_DIR OUT_DIR NUM_JOBS [-x EXCLUDED_SPANS]"

if [ "$#" -lt 5 ]; then
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
    | awk 'BEGIN{OFS="\t"}{print $1,$2,$3,"1.0"}' \
    > $gtab
gold=$outdir/gold.combined.tsv
./nel prepare-tac -q $goldx $gtab $@ \
    > $gold


# CONVERT SYSTEMS TO EVALUATION FORMAT
echo "INFO Converting systems to evaluation format.."
ls $sysdir/*.tab \
    | xargs -I{} -n 1 -P $jobs $SCR/run_tac14_prepare.sh $outdir {} $@


# EVALUATE
echo "INFO Evaluating systems.."
ls $outdir/*.combined.tsv \
    | grep -v "gold\.combined\.tsv$" \
    | xargs -n 1 -P $jobs $SCR/run_evaluate.sh $gold


# PREPARE SUMMARY REPORT
echo "INFO Preparing summary report.."
$SCR/run_tac14_report.sh $outdir

