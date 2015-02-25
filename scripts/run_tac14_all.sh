#!/usr/bin/env bash
#
# Run TAC14 evaluation and analysis
set -e

usage="Usage: $0 GOLD_XML GOLD_TAB SYSTEMS_DIR OUT_DIR [-x EXCLUDED_SPANS]"

if [ "$#" -lt 4 ]; then
    echo $usage
    exit 1
fi

goldx=$1; shift # gold standard queries/mentions (XML)
goldt=$1; shift # gold standard link annotations (tab-separated)
sysdir=$1; shift  # directory containing output from systems
outdir=$1; shift # directory to which results are written

SCR=`dirname $0`

# number of jobs for parallel mode (set to number of CPUs if possible)
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null) || JOBS=8
FMT='tab' # format for confidence and significance output ('tab' or 'json')
CONFIDENCE_MEASURES=(
    strong_mention_match
    strong_typed_mention_match
    strong_link_match
    strong_typed_link_match
    entity_match
)


# CALCULATE SCORES
$SCR/run_tac14_evaluation.sh $goldx $goldt $sysdir $outdir $JOBS $@

# CALCULATE COMPOSITE SCORES
./nel compose-measures -r strong_all_match strong_mention_match -r strong_typed_mention_match strong_mention_match $outdir/*.evaluation

# GET GOLD STANDARD PATH
gold=$outdir/gold.combined.tsv
if [ ! -e $gold ]
then
    echo "ERROR $gold does not exist"
    exit 1
fi


# GET LIST OF SYSTEM OUTPUT PATHS
systems=(`ls $outdir/*.combined.tsv | grep -v "gold\.combined\.tsv$"`)
if [ ${#systems[*]} == 0 ]
then
    echo "ERROR did not find any system output"
    exit 1
fi


## CALCULATE METRIC CORRELATIONS
#if [ ${#systems[*]} != 1 ]
#then
#	echo "INFO calculating measure correlations"
#	./nel compare-measures -e -m tac14 -f plot -s eigen --out-fmt $outdir/00measure-{}.pdf $outdir/*.evaluation
#fi

# CALCULATE CONFIDENCE INTERVALS
echo "INFO Calculating confidence intervals.."
for sys in ${systems[@]}
do
    conf=`echo $sys | sed 's/\.combined.tsv/.confidence/'`
    ./nel confidence \
        -m all-tagging \
        -f tab \
        -g $gold \
        -j $JOBS \
        $sys \
        > $conf
done


# PREPARE SUMMARY CONFIDENCE INTERVAL REPORTS
$SCR/run_report_confidence.sh $outdir ${CONFIDENCE_MEASURES[@]}


# CALCULATE ALL PAIRWISE SIGNIFICANCE TESTS (NOTE: THIS TAKES A LITTLE WHILE)
#echo "INFO Calculating significance.."
#sign=$outdir/00report.significance.$FMT
#./nel significance \
#    -g $gold \
#    --permute \
#    -j $JOBS \
#    -f $FMT \
#    ${systems[@]} \
#    > $sign


# RUN ERROR ANALYSIS
echo "INFO Preparing error report.."
printf "%s\n" "${systems[@]}" \
    | xargs -n 1 -P $JOBS $SCR/run_analysis.sh $gold

