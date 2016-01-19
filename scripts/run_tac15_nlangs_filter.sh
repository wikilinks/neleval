#!/usr/bin/env bash
set -e

OUT_DIR="$1"
NJOBS=$2
SCR=`dirname $0`

if [ -z $OUT_DIR ]
then
    echo 'Usage: '$0 OUT_DIR [N_JOBS] >&2
    exit 1
fi
if [ -z $NJOBS ]
then
    NJOBS=1
fi

# Calculate number of languages for each gold entity:

f=$(mktemp)
cat $OUT_DIR/gold.combined.tsv | cut -f 1,4 | sed 's/^\(...\)[^[:space:]]*/\1/' | sort -u | cut -f2 | sort | uniq -c > $f
n=1
for d in $OUT_DIR/00monolingual $OUT_DIR/00trilingual
do
    echo ==== Evaluating $d ====
    mkdir $d
    cp $OUT_DIR/*.combined.tsv $d
    echo Filtering
    grep -wFf <(cat $f | awk '$1 == '$n' { print $2 }') $OUT_DIR/gold.combined.tsv > $d/gold.combined.tsv
    echo Evaluating over $(cat $d/gold.combined.tsv | wc -l) of $(cat $OUT_DIR/gold.combined.tsv | wc -l) gold mentions
    ls $d/*.combined.tsv \
        | grep -v "gold\.combined\.tsv$" \
        | xargs -n 1 -P $NJOBS $SCR/run_evaluate.sh $d/gold.combined.tsv
    n=3
done
rm $f
