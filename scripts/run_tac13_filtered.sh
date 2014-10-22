#!/usr/bin/env bash
#
# Run TAC13 filtered evaluation and analysis
set -e

usage="Usage: $0 GOLD_XML GOLD_TAB SYSTEMS_DIR OUT_DIR"

if [ "$#" -ne 4 ]; then
    echo $usage
    exit 1
fi

goldx=$1; shift # gold standard queries/mentions (XML)
goldt=$1; shift # gold standard link annotations (tab-separated)
sysdir=$1; shift  # directory containing output from systems
outdir=$1; shift # directory to which results are written

SCR=`dirname $0`

JOBS=8 # number of jobs for parallel mode (set to number of CPUs if possible)


# CONFIGURE FILTERS
FILTERS=(
    # NE type filters
    "PER:::PER$"
    "ORG:::ORG$"
    "GPE:::GPE$"
    # genre filters
    "NW:::^(AFP|APW|CNA|LTW|NYT|WPB|XIN)_ENG_"
    "WB:::^eng-(NG|WL)-"
    "DF:::^bolt-eng-DF-"
    # combined filters
    "PER_NW:::^(AFP|APW|CNA|LTW|NYT|WPB|XIN)_ENG_.*PER$"
    "PER_WB:::^eng-(NG|WL)-.*PER$"
    "PER_DF:::^bolt-eng-DF-.*PER$"
    "ORG_NW:::^(AFP|APW|CNA|LTW|NYT|WPB|XIN)_ENG_.*ORG$"
    "ORG_WB:::^eng-(NG|WL)-.*ORG$"
    "ORG_DF:::^bolt-eng-DF-.*ORG$"
    "GPE_NW:::^(AFP|APW|CNA|LTW|NYT|WPB|XIN)_ENG_.*GPE$"
    "GPE_WB:::^eng-(NG|WL)-.*GPE$"
    "GPE_DF:::^bolt-eng-DF-.*GPE$"
    )


# RUN OVERALL EVALUATION
$SCR/run_tac13_evaluation.sh $goldx $goldt $sysdir $outdir $JOBS


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
        $SCR/run_tac13_report.sh $subdir

    else
        echo "WARN Ignoring filter '$regex' - no gold mentions"
    fi

done
