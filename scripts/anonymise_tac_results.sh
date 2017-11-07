#!/bin/bash
###set -x

EXTS=".evaluation .combined.tsv .eval_prefixed .evaluation_bydoc .confidence"
CMD=$0
SORT_MEASURE=$1
shift
OUT_DIR=$1
shift
MAIN_DIR=$1
shift
ALL_DIRS="$MAIN_DIR $@"
if [ -z "$MAIN_DIR" ]
then
	echo "USAGE: $CMD SORT_MEASURE OUT_DIR IN_DIR ..." >&2
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


top=$(python -m neleval rank-systems --group-limit 1 $TAC_GROUPRE -m $SORT_MEASURE $SORT_EVALS | tail -n+2 | cut -f6)
seq=$(seq 1 100)
mapping=$(paste <(echo $seq | tr " " "\n") <(echo $top | tr " " "\n") | grep -E '\t.')
echo "$mapping"
sub=$(echo "$mapping" | sed 's/^\(.*\)	\(.*\)/s\/^\2\/\1\//' | tr '\n' ';')
echo $sub
for f in $(find $ALL_DIRS $(echo " "$EXTS | sed 's/ / -o -name */g;s/^ -o//'))
do
	o=$OUT_DIR/$(dirname $f)/$(basename $f | sed "$sub")
	mkdir -p $(dirname "$o")
	ln -s "$PWD/$f" "$o"
done
