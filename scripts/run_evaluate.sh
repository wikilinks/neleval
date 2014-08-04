#!/usr/bin/env bash
#
# Evaluate and save to file

usage="Usage: $0 GOLD SYSTEM"

if [ "$#" -ne 2 ]; then
    echo $usage
    exit 1
fi

gold=$1; shift # prepared gold standard annotations (.combined.tsv)
sys=$1; shift # prepared system annotations (.combined.tsv)

out=`echo $sys | sed 's/.combined.tsv/.evaluation/'`
./nel evaluate \
    -l tac \
    -c tac \
    -f 'tab_format' \
    -g $gold \
    $sys \
    > $out
