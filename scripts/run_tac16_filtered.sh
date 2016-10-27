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
    "NAM:::/NAM$"
    "NOM:::/NOM$"
    # language filters
    "CMN:::^CMN"
    "ENG:::^(NYT_ENG|ENG)"
    "SPA:::^SPA"

    # genre filters
    "NW:::^(..._NW|NYT_ENG)"
    "DF:::^..._DF"
    # combined with language
    "CMN_NW:::^CMN_NW"
    "CMN_DF:::^CMN_DF"
    "CMN_FAC:::^CMN.*FAC/N..$"
    "CMN_GPE:::^CMN.*GPE/N..$"
    "CMN_LOC:::^CMN.*LOC/N..$"
    "CMN_ORG:::^CMN.*ORG/N..$"
    "CMN_PER:::^CMN.*PER/N..$"
    "CMN_NAM:::^CMN.*/NAM$"
    "CMN_NOM:::^CMN.*/NOM$"
    "ENG_NW:::^(ENG_NW|NYT_ENG)"
    "ENG_DF:::^ENG_DF"
    "ENG_FAC:::^(ENG|NYT_ENG).*FAC/N..$"
    "ENG_GPE:::^(ENG|NYT_ENG).*GPE/N..$"
    "ENG_LOC:::^(ENG|NYT_ENG).*LOC/N..$"
    "ENG_ORG:::^(ENG|NYT_ENG).*ORG/N..$"
    "ENG_PER:::^(ENG|NYT_ENG).*PER/N..$"
    "ENG_NAM:::^(ENG|NYT_ENG).*/NAM$"
    "ENG_NOM:::^(ENG|NYT_ENG).*/NOM$"
    "SPA_NW:::^SPA_NW"
    "SPA_DF:::^SPA_DF"
    "SPA_FAC:::^SPA.*FAC/N..$"
    "SPA_GPE:::^SPA.*GPE/N..$"
    "SPA_LOC:::^SPA.*LOC/N..$"
    "SPA_ORG:::^SPA.*ORG/N..$"
    "SPA_PER:::^SPA.*PER/N..$"
    "SPA_NAM:::^SPA.*/NAM$"
    "SPA_NOM:::^SPA.*/NOM$"

    # combined with genre
    "NW_FAC:::^(..._NW|NYT_ENG).*FAC/N..$"
    "NW_GPE:::^(..._NW|NYT_ENG).*GPE/N..$"
    "NW_LOC:::^(..._NW|NYT_ENG).*LOC/N..$"
    "NW_ORG:::^(..._NW|NYT_ENG).*ORG/N..$"
    "NW_PER:::^(..._NW|NYT_ENG).*PER/N..$"
    "NW_NAM:::^(..._NW|NYT_ENG).*/NAM$"
    "NW_NOM:::^(..._NW|NYT_ENG).*/NOM$"
    "DF_FAC:::^..._DF.*FAC/N..$"
    "DF_GPE:::^..._DF.*GPE/N..$"
    "DF_LOC:::^..._DF.*LOC/N..$"
    "DF_ORG:::^..._DF.*ORG/N..$"
    "DF_PER:::^..._DF.*PER/N..$"
    "DF_NAM:::^..._DF.*/NAM$"
    "DF_NOM:::^..._DF.*/NOM$"

    "ENG_ORG,PER,GPE_NAM:::^(ENG|NYT_ENG).*(ORG|PER|GPE)/NAM$"
    "SPA_ORG,PER,GPE_NAM:::^SPA.*(ORG|PER|GPE)/NAM$"
    "CMN_ORG,PER,GPE_NAM:::^CMN.*(ORG|PER|GPE)/NAM$"
    "ENG_ORG,PER,GPE_NOM:::^(ENG|NYT_ENG).*(ORG|PER|GPE)/NOM$"
    "SPA_ORG,PER,GPE_NOM:::^SPA.*(ORG|PER|GPE)/NOM$"
    "CMN_ORG,PER,GPE_NOM:::^CMN.*(ORG|PER|GPE)/NOM$"
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
