#!/usr/bin/env bash

#
# Run TAC13 filtered evaluation and analysis
set -e

usage="Usage: $0 OUTDIR REGEX GOLD SYSTEM"

if [ "$#" -ne 4 ]; then
    echo $usage
    exit 1
fi

outdir=$1; shift # directory to which results are written
regex=$1; shift # POSIX 1003.2 regular expression for filtering
gold=$1; shift # prepared gold standard annotations (.combined.tsv)
sys=$1; shift # prepared system annotations (.combined.tsv)

SCR=`dirname $0`

FMT='tab' # format for confidence and significance output ('tab' or 'json')


# FILTER GOLD STANDARD
goldf=$outdir/`basename $gold`
cat $gold \
    | egrep "$regex" \
    > $goldf


# FILTER SYSTEM OUTPUT
sysf=$outdir/`basename $sys`
cat $sys \
    | egrep "$regex" \
    > $sysf


# EVALUATE ON FILTERED SUBSET
out=`echo $sysf | sed 's/.combined.tsv/.evaluation/'`
./nel evaluate \
    -m all \
    -f $FMT \
    -g $goldf \
    $sysf \
    > $out
