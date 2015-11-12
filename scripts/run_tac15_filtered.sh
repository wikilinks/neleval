#!/usr/bin/env bash
#
# Run TAC14 filtered evaluation and analysis
set -e

usage="Usage: $0 OUT_DIR\\n\\nOUT_DIR should include *.combined.tsv outputs from an evaluation script"

if [ "$#" -ne 1 ]; then
    echo -e $usage
    exit 1
fi

outdir=$1; shift # directory to which results are written

SCR=`dirname $0`

JOBS=2 # number of jobs for parallel mode (set to number of CPUs if possible)


# CONFIGURE FILTERS
FILTERS=(
    # NE type filters
    "FAC:::FAC/N..$"
    "GPE:::GPE/N..$"
    "LOC:::LOC/N..$"
    "ORG:::ORG/N..$"
    "PER:::PER/N..$"
    "PERNAM:::PER/NAM$"
    "PERNOM:::PER/NOM$"
    "NAM:::/NAM$"
    # language filters
    "CMN:::^CMN"
    "ENG:::^ENG"
    "SPA:::^SPA"
    # genre filters
    "NW:::^..._NW"
    "DF:::^..._DF"
    # combined with language
    "CMN_NW:::^CMN_NW"
    "CMN_DF:::^CMN_DF"
    "CMN_FAC:::^CMN.*FAC/N..$"
    "CMN_GPE:::^CMN.*GPE/N..$"
    "CMN_LOC:::^CMN.*LOC/N..$"
    "CMN_ORG:::^CMN.*ORG/N..$"
    "CMN_PER:::^CMN.*PER/N..$"
    "ENG_NW:::^ENG_NW"
    "ENG_DF:::^ENG_DF"
    "ENG_FAC:::^ENG.*FAC/N..$"
    "ENG_GPE:::^ENG.*GPE/N..$"
    "ENG_LOC:::^ENG.*LOC/N..$"
    "ENG_ORG:::^ENG.*ORG/N..$"
    "ENG_PER:::^ENG.*PER/N..$"
    "ENG_PERNAM:::^ENG.*PER/NAM$"
    "ENG_PERNOM:::^ENG.*PER/NOM$"
    "ENG_NAM:::^ENG.*/NAM$"
    "SPA_NW:::^SPA_NW"
    "SPA_DF:::^SPA_DF"
    "SPA_FAC:::^SPA.*FAC/N..$"
    "SPA_GPE:::^SPA.*GPE/N..$"
    "SPA_LOC:::^SPA.*LOC/N..$"
    "SPA_ORG:::^SPA.*ORG/N..$"
    "SPA_PER:::^SPA.*PER/N..$"

    # combined with genre
    "NW_FAC:::^..._NW.*FAC/N..$"
    "NW_GPE:::^..._NW.*GPE/N..$"
    "NW_LOC:::^..._NW.*LOC/N..$"
    "NW_ORG:::^..._NW.*ORG/N..$"
    "NW_PER:::^..._NW.*PER/N..$"
    "NW_PERNAM:::^..._NW.*PER/NAM$"
    "NW_PERNOM:::^..._NW.*PER/NOM$"
    "NW_NAM:::^..._NW.*/NAM$"
    "DF_FAC:::^..._DF.*FAC/N..$"
    "DF_GPE:::^..._DF.*GPE/N..$"
    "DF_LOC:::^..._DF.*LOC/N..$"
    "DF_ORG:::^..._DF.*ORG/N..$"
    "DF_PER:::^..._DF.*PER/N..$"
    "DF_PERNAM:::^..._DF.*PER/NAM$"
    "DF_PERNOM:::^..._DF.*PER/NOM$"
    "DF_NAM:::^..._DF.*/NAM$"

    "ORG,PER,GPE:::^ENG.*(ORG|PER|GPE)/NAM$"
    )


# GET GOLD STANDARD PATH
gold=$outdir/gold.combined.tsv
if [ ! -e $gold ]
then
    echo "ERROR $gold does not exist"
    exit 1
fi


# GET LIST OF SYSTEM OUTPUT PATHS
systems=(`ls $outdir/*.combined.tsv | grep -v "gold\.combined\.tsv$"`)
if [ ${#systems[*]} == 0 ]
then
    echo "ERROR did not find any system output"
    exit 1
fi


# RUN FILTERED EVALUTION
for filter in ${FILTERS[@]}
do
    subset=`echo $filter | sed 's/:::.*$//'`
    regex=`echo $filter | sed 's/^.*::://'`

    # MAKE DIRECTORY FOR FILTERED EVALUATION
    subdir=$outdir/00filtered/$subset
    mkdir -p $subdir

    # FILTER GOLD STANDARD
    goldf=$subdir/`basename $gold`
    cat $gold \
        | egrep "$regex" \
        > $goldf

    if [ -s $goldf ]
    then

        # FILTER AND EVALUATE SYSTEMS
        echo "INFO Evaluating on $subset subset.."
        printf "%s\n" "${systems[@]}" \
            | xargs -n 1 -P $JOBS $SCR/run_filtrate.sh $subdir "$regex" $goldf

        # PREPARE SUMMARY REPORT
        echo "INFO Preparing summary report.."
        $SCR/run_tac14_report.sh $subdir

    else
        echo "WARN Ignoring filter '$regex' - no gold mentions"
    fi

done
