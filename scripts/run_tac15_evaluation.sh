#!/usr/bin/env bash
set -e

usage="Usage: $0 GOLD SYSTEMS_DIR OUT_DIR NUM_JOBS [-x EXCLUDED_SPANS]"

if [ "$#" -lt 4 ]; then
    echo $usage
    exit 1
fi

cleanup_cmd='OFS="\t" {} $6 == "TTL/NAM" {$6 = "PER/NOM"} ($6 ~ /NAM$/ || ($6 ~ /PER.NOM$/ && $1 ~ /^ENG/)) && $1 != "CMN_NW_001331_20150702_F00100023"'

gtab=$1; shift # gold standard link annotations (tab-separated)
sysdir=$1; shift # directory containing output from systems
outdir=$1; shift # directory to which results are written
jobs=$1; shift # number of jobs for parallel mode

SCR=`dirname $0`


# CONVERT GOLD TO EVALUATION FORMAT
echo "INFO Converting gold to evaluation format.."
# XXX: "combined" is a misnomer in tac15. Should be neleval? But existing scripts depend on this extension.
gold=$outdir/gold.combined.tsv
options=$@
./nel prepare-tac15 $gtab $options | awk -F'\t' "$cleanup_cmd" > $gold

# convert systems to evaluation format
echo "INFO converting systems to evaluation format.."
ls $sysdir/* \
	| xargs -I{} -n 1 -P $jobs bash -c "f={}; ./nel prepare-tac15 \$f $options | awk -F'\t' '$cleanup_cmd' > $outdir/\$(basename \$f).combined.tsv"



# EVALUATE
echo "INFO Evaluating systems.."
ls $outdir/*.combined.tsv \
    | grep -v "gold\.combined\.tsv$" \
    | xargs -n 1 -P $jobs $SCR/run_evaluate.sh $gold


# PREPARE SUMMARY REPORT
echo "INFO Preparing summary report.."
$SCR/run_tac14_report.sh $outdir

