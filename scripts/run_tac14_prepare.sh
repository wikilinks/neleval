#!/usr/bin/env bash
#
# Convert TAC system output to evaluation format
set -e

usage="Usage: $0 OUT_DIR SYSTEM_TAB [-x EXCLUDED_SPANS]"

if [ "$#" -lt 2 ]; then
    echo $usage
    exit 1
fi

outdir=$1; shift # output directory
syst=$1; shift # system link annotations (tab-separated)
               # mentions file must have same path but with .xml extension

# FIND SYSTEM XML CORRESPONDING TO GIVEN TAB FILE
pre=`echo $syst | sed 's/\.tab//'`
sysx=$pre.xml
if [ ! -e $sysx ]
then
    echo "ERROR could not find xml for $syst"
    exit 1
fi


# RUN PREPARE SCRIPT
out=$outdir/`basename $pre`.combined.tsv
./nel prepare-tac -q $sysx $syst $@ \
    > $out
