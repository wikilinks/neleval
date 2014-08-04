#!/usr/bin/env bash
#
# Run analysis and save to file

usage="Usage: $0 GOLD_XML GOLD_TAB SYSTEMS_DIR OUT_DIR"

if [ "$#" -ne 2 ]; then
    echo $usage
    exit 1
fi

gold=$1; shift # prepared gold standard annotations (.combined.tsv)
sys=$1; shift # prepared system annotations (.combined.tsv)

out=`echo $sys | sed 's/.combined.tsv/.analysis/'`
./nel analyze \
    -s \
    -g $gold \
    $sys \
    > $out
