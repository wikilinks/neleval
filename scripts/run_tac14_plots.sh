#!/usr/bin/env bash
#
# Generate many plots
set -e

usage="Usage: $0 EVALUATION_OUT_DIR"

if [ "$#" -ne 1 ]; then
    echo $usage
    exit 1
fi

evaldir=$1; shift # directory to which results were
# strip trailing slash
evaldir="${evaldir%/}"
plotdir=$evaldir/plots

echo Putting plots in $plotdir

mkdir -vp $plotdir/columns/{all,team-best}/90ci $plotdir/scatter/ $plotdir/by-system $plotdir/by-team $plotdir/heatmap $plotdir/measure-cmp

ALL_MEASURES=$(cat $(ls $evaldir/*.evaluation | head -n1) | awk 'NR > 1 {print $8}' | sort | sed 's/^/-m /')
CONF_MEASURES=$(cat $(ls $evaldir/*.confidence | head -n1) | awk 'NR > 1 {print $1}' | sort | sed 's/^/-m /')
GROUPRE="--group-re=(?<=/)[^/]*(?=[0-9]\.)"

# TODO: Determine figure sizes dynamically
square_figsize="6,6"
syscols_figsize="17,4"
teamcols_figsize="8,4"

./nel plot-systems --by-measure --scatter -o $plotdir/scatter/{}.pdf $GROUPRE $ALL_MEASURES --figsize=$square_figsize --sort-by=name $evaldir/*.evaluation
for d in all team-best
do
	if [ $d == "team-best" ]
	then
		arg="--best-in-group $GROUPRE"
		nociarg="--line --prf"
		figsize=$teamcols_figsize
	else
		arg=
		nociarg=
		figsize=$teamcols_figsize
		figsize=$syscols_figsize
	fi
	./nel plot-systems --by-measure --columns -o $plotdir/columns/$d/{}.pdf $ALL_MEASURES --figsize=$figsize $nociarg $arg --sort-by=score $evaldir/*.evaluation
	./nel plot-systems --by-measure --columns -o $plotdir/columns/$d/90ci/{}.pdf $CONF_MEASURES -i confidence --ci=90 $arg --figsize=$figsize --sort-by=score $evaldir/*.confidence
done

for d in by-system by-team
do
	if [ d == "by-team" ]
	then
		arg=$GROUPRE
	else
		arg=
	fi
	# XXX: Should this just show TAC measures?
	./nel plot-systems --by-system --scatter -o $plotdir/$d/{}.pdf $arg $ALL_MEASURES --figsize=$square_figsize --sort-by name $evaldir/*.evaluation
done

# XXX: should sort be by mention_ceaf?
./nel plot-systems --heatmap --by-measure -o $plotdir/heatmap/all.pdf --sort-by=name --figsize=$syscols_figsize $ALL_MEASURES $evaldir/*.evaluation
./nel plot-systems --heatmap --by-measure -o $plotdir/heatmap/tac14.pdf --sort-by=name -m tac14 --figsize=$syscols_figsize $evaldir/*.evaluation
./nel plot-systems --heatmap --by-measure -o $plotdir/heatmap/tac14-official.pdf --sort-by=name --figsize=$syscols_figsize -m strong_mention_match -m strong_typed_mention_match -m strong_all_match -m mention_ceaf $evaldir/*.evaluation

./nel compare-measures -e $ALL_MEASURES -f plot -s eigen --out-fmt $plotdir/measure-cmp/all-{}.pdf $evaldir/*.evaluation
./nel compare-measures -e -m tac14 -f plot -s name --out-fmt $plotdir/measure-cmp/tac14-{}.pdf $evaldir/*.evaluation
