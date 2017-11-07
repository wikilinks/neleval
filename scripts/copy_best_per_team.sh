#!/bin/bash

CMD=$0
SORT_MEASURE=$1
shift
OUT_DIR=$1
shift
MAIN_DIR=$1
shift
ALL_DIRS="$MAIN_DIR $@"
if [ -z "$OUT_DIR" ]
then
    echo "Usage: $CMD SORT_MEASURE OUT_DIR MAIN_DIR ..." >&2
    exit 1
fi

SORT_EVALS=$MAIN_DIR/*.evaluation
if echo $SORT_EVALS | grep -q '/[0-9]+\.evaluation'
then
	TAC_GROUPRE="--group-re=(?<=/)[^/]*(?=\.)"
elif echo $SORT_EVALS | grep -q '[0-9][0-9]*\.evaluation'
then
	TAC_GROUPRE="--group-re=(?<=/)[^/]*(?=[0-9][0-9]*\.)"
else
	TAC_GROUPRE="--group-re=(?<=/)[^/]*(?=[0-9]\.)"
fi


top=$(python -m neleval rank-systems $TAC_GROUPRE -m $SORT_MEASURE --group-limit 1 $SORT_EVALS | tail -n+2 | cut -f7 | sed 's|.*/||' | cut -d. -f1)
groups=$(python -m neleval rank-systems $TAC_GROUPRE -m $SORT_MEASURE --group-limit 1 $SORT_EVALS | tail -n+2 | cut -f6)

sed_cmd=$(paste <(echo "$top") <(echo "$groups") | sed 's/^\(.*\)	\(.*\)$/s|^\1\\.|\2.|;/')
###echo "$sed_cmd"

for f in $(find $ALL_DIRS -false $(echo "$top" | sed 's/.*/-o -name &.*/'))
do
    o="$OUT_DIR/$(dirname "$f")/$(basename "$f" | sed "$sed_cmd")"
    mkdir -p $(dirname $o)
    ln -s $PWD/$f $o
done
