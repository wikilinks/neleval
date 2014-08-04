#!/usr/bin/env bash
#
# Convert TAC system output to evaluation format
set -e

usage="Usage: $0 GOLD_XML GOLD_TAB SYSTEMS_DIR OUT_DIR"

if [ "$#" -ne 3 ]; then
    echo $usage
    exit 1
fi

goldx=$1; shift # gold standard queries/mentions (XML)
outdir=$1; shift # output directory
syst=$1; shift # system link annotations (tab-separated)

sys=`basename $syst`
stab=$outdir/$sys.tab
cat $syst \
    | awk 'BEGIN{OFS="\t"} {print $1,$2,"NA",$3}' \
    > $stab
#rm $stab
out=$outdir/$sys.combined.tsv
./nel prepare-tac \
    -q $goldx \
    $stab \
    > $out
