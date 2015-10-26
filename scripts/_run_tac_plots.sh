usage="Usage: $0 EVALUATION_OUT_DIR"

if [ "$#" -ne 1 ]; then
    echo $usage
    exit 1
fi

evaldir=$1; shift # directory to which results were
# strip trailing slash
evaldir="${evaldir%/}"
plotdir=$evaldir/00plots

echo Putting plots in $plotdir

mkdir -vp $plotdir/columns/{all,team-best}/90ci $plotdir/scatter/ $plotdir/by-system $plotdir/by-team $plotdir/single/{heatmap,plot} $plotdir/measure-cmp

ALL_MEASURES=$(cat $(ls $evaldir/*.evaluation | head -n1) | awk 'NR > 1 && $0 !~ /\// {print $8}' | sort | sed 's/^/-m /')
CONF_MEASURES=$(cat $(ls $evaldir/*.confidence | head -n1) | awk 'NR > 1 && $0 !~ /\// {print $1}' | sort | sed 's/^/-m /')
GROUPRE="--group-re=(?<=/)[^/]*(?=[0-9]\.)"
LABELMAP='--label-map={"fscore": "$F_1$", "precision": "$P$", "recall": "$R$", "b_cubed_plus": "B-Cubed+", "b_cubed": "B-Cubed", "strong_mention_match": "NER", "strong_typed_mention_match": "NERC", "strong_all_match": "NERL", "mention_ceaf": "CEAFm", "entity_ceaf": "CEAFe", "entity_match": "KB ID sets"}'

./nel plot-systems "$LABELMAP" --by-measure --scatter -o $plotdir/scatter/{}.pdf $GROUPRE $ALL_MEASURES --figsize=$square_figsize --sort-by=name $evaldir/*.evaluation
for d in all team-best
do
	if [ $d == "team-best" ]
	then
		arg="--best-in-group $GROUPRE"
		nociarg="--lines --prf"
		figsize=$teamcols_figsize
	else
		arg=
		nociarg=
		figsize=$teamcols_figsize
		figsize=$syscols_figsize
	fi
	./nel plot-systems "$LABELMAP" --by-measure --columns -o $plotdir/columns/$d/{}.pdf $ALL_MEASURES --figsize=$figsize $nociarg $arg --sort-by=score $evaldir/*.evaluation
	./nel plot-systems "$LABELMAP" --by-measure --columns -o $plotdir/columns/$d/90ci/{}.pdf $CONF_MEASURES -i confidence --ci=90 $arg --figsize=$figsize --sort-by=score $evaldir/*.confidence
done

for d in by-system by-team
do
	if [ $d == "by-team" ]
	then
		arg=$GROUPRE
	else
		arg=
	fi
	# XXX: Should this just show TAC measures?
	./nel plot-systems "$LABELMAP" --by-system --scatter -o $plotdir/$d/{}.pdf $arg $ALL_MEASURES --figsize=$square_figsize --sort-by name $evaldir/*.evaluation
done

./nel plot-systems "$LABELMAP" --heatmap --by-measure -o $plotdir/single/heatmap/all.pdf --sort-by=name --figsize=$syscols_figsize $ALL_MEASURES --sort-by=$DEFAULT_MEASURE $evaldir/*.evaluation
./nel plot-systems "$LABELMAP" --heatmap --by-measure -o $plotdir/single/heatmap/official.pdf --sort-by=name --figsize=$syscols_figsize $OFFICIAL_MEASURES --sort-by=$DEFAULT_MEASURE $evaldir/*.evaluation

./nel plot-systems "$LABELMAP" --single-plot -o $plotdir/single/plot/official.pdf --sort-by=name --figsize=$syscols_figsize $OFFICIAL_MEASURES --sort-by=$DEFAULT_MEASURE --line $evaldir/*.evaluation

./nel compare-measures "$LABELMAP" -e $ALL_MEASURES -f plot -s eigen --out-fmt $plotdir/measure-cmp/all-{}.pdf $evaldir/*.evaluation
./nel compare-measures "$LABELMAP" -e -m tac14 -f plot -s name --out-fmt $plotdir/measure-cmp/tac14-{}.pdf $evaldir/*.evaluation
