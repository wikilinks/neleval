#!/usr/bin/env bash
#
# Run TAC15 weak evaluation
set -e

usage="Usage: $0 OUT_DIR\\n\\nOUT_DIR should include *.combined.tsv outputs from an evaluation script"

if [ "$#" -ne 1 ]; then
    echo -e $usage
    exit 1
fi

outdir=$1; shift # directory to which results are written
subdir=$outdir/00weak
mkdir -p $subdir

SCR=`dirname $0`

JOBS=1 # number of jobs for parallel mode (set to number of CPUs if possible)

# CONVERT GOLD TO CHAR-LEVEL ANNOTATIONS
gold=$outdir/gold.combined.tsv
if [ ! -e $gold ]
then
    echo "ERROR $gold does not exist"
    exit 1
fi
gweak=$subdir/gold.combined.tsv
./nel to-weak $gold > $gweak

# GET LIST OF SYSTEM OUTPUT PATHS
systems=(`ls $outdir/*.combined.tsv | grep -v "gold\.combined\.tsv$"`)
if [ ${#systems[*]} == 0 ]
then
    echo "ERROR did not find any system output"
    exit 1
fi

# RUN WEAK EVALUTION
for sys in ${systems[@]}
do
    echo $sys
    sweak=$subdir/`basename $sys`
    ./nel to-weak $sys > $sweak

    out=`echo $sweak | sed 's/.combined.tsv/.evaluate/'`
    ./nel evaluate \
        -m all-tagging \
        -f 'tab' \
        -g $gweak \
        $sweak \
        > $out
done
