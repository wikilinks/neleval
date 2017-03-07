#!/usr/bin/env bash

usage="Usage: $0 [-b best_fmt] -m MEASURE [-m MEASURE ...] SYTEM [SYSTEM ...]"
source $(dirname $0)/_common.sh

if [ "$#" -lt 3 ]; then
    echo $usage
    exit 1
fi

best_fmt="%s"
mult=1
dp=3
measures=
while getopts :m:b:p opt
do
    case $opt in
    m)
        measures="$measures $OPTARG"
        ;;
    b)
        best_fmt="$OPTARG"
        ;;
    p)
        mult=100
        dp=1
        ;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    esac
done
shift $((OPTIND - 1))

best_fmt=$(printf $best_fmt "%0.${dp}f")

paths="$@"

header() {
    echo System
    while [ -n "$1" ]
    do
        echo $1 P
        echo $1 R
        echo $1 F
        shift
    done
}

header $measures | tr "\n" "\t" | sed 's/.$//g'
tmpf=$(mktemp $TMPDIR/XXXXX)
for eval in $paths
do
    scores=$(cat $eval | get_eval_prf $measures)
    if echo $scores | grep -q [1-9]
    then
        # not all scores are zero
        echo $(basename $eval | sed 's/\.[^.]*$//')'	' "$scores"
    fi
done > $tmpf
awk -F'\t' '
FNR == 1 {
    filenum += 1
}
filenum == 1 {
    for (i=2; i <= NF; i++) {
        if (best[i] <= 1.0 * $i) {
            best[i] = 1.0 * $i;
        } 
    }
}
filenum == 2 {
    OFS="\t"
    for (i=2; i <= NF; i++) {
        if ($i > best[i] - 0.00001) {
            $i = sprintf("'$best_fmt'", $i * '$mult')
        } else {
            $i = sprintf("%0.'$dp'f", $i * '$mult')
        }
    }
    print
}
' $tmpf $tmpf
rm $tmpf
