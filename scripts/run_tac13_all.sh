#!/usr/bin/env bash
#
# Run TAC13 evaluation and analysis

usage="Usage: $0 GOLD_XML GOLD_TAB SYSTEMS_DIR OUT_DIR"

if [ "$#" -ne 4 ]; then
    echo $usage
    exit 1
fi

goldx=$1; shift # gold standard queries/mentions (XML)
goldt=$1; shift # gold standard link annotations (tab-separated)
sysdir=$1; shift  # directory containing output from systems
outdir=$1; shift # directory to which results are written

SCR=`dirname $0`

TRIALS=10000 # number of trials for significance (10000 or smaller for testing)
JOBS=16 # number of jobs for parallel mode
FMT='tab' # format for significance output ('tab' or 'json')


# CALCULATE SCORES
$SCR/run_tac13_evaluation.sh $goldx $goldt $sysdir $outdir $JOBS
gold=$outdir/gold.combined.tsv
if [ ! -e $gold ]
then
    echo "ERROR $gold does not exist"
    exit
fi
systems=(`ls $outdir/*.combined.tsv | grep -v "gold\.combined\.tsv$"`)
if [ ${#systems[*]} == 0 ]
then
    echo "ERROR did not find any system output"
fi


# CALCULATE ALL PAIRWISE SIGNIFICANCE TESTS (NOTE: THIS TAKES A LITTLE WHILE)
echo "INFO Calculating significance.."
sign=$outdir/00report.significance.$FMT
./nel significance \
    -g $gold \
    -n $TRIALS \
    --permute \
    -j $JOBS \
    -f $FMT \
    ${systems[@]} \
    > $sign


# RUN ERROR ANALYSIS
echo "INFO Preparing error report.."
printf "%s\n" "${systems[@]}" \
    | xargs -n 1 -P $JOBS $SCR/run_analysis.sh $gold


# TODO EVALUATE BY ENTITY TYPE
# TODO EVALUATE BY DOCUMENT GENRE
# TODO EVALUATE BY DOCUMENT GENRE AND ENTITY TYPE

