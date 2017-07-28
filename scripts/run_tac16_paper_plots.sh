#!/bin/bash
# XXX: THIS SHOULD REALLY BE A MAKEFILE
set +x
set -e
EDL_DIR=$1
PLOTS_DIR=$2
if [ -z "$PLOTS_DIR" ]
then
    echo Usage: EDL_DIR' PLOTS_DIR'
    exit 1
fi
if [ ! -d "$PLOTS_DIR" ]
then
    echo please create '$PLOTS_DIR'
    exit 1
fi

ep_ext() {
    sed 's|\.evaluation|\.eval_prefixed|g'
}

get_top() {
    n=$1
    shift
    m=$1
    shift
    systems=$@
    ./nel rank-systems $TAC_GROUPRE -m $m --max $n --group-limit 1 $systems | tail -n+2 | cut -f7
}

ALL_EDL=$EDL_DIR/*.evaluation
./nel compose-measures -r strong_all_match strong_mention_match -r strong_typed_mention_match strong_mention_match $ALL_EDL
tmpd=$(mktemp -d /tmp/weakXXXXXX)
###d=$EDL_DIR-weakmatch; ./scripts/merge_evaluations.py --label-re='[^/]+/?$' --out-extension evaluation -l =$(basename $d) --out-dir $tmpd $d $(find $d/00filtered/ -type d )
d=$EDL_DIR; ./scripts/merge_evaluations.py --label-re='[^/]+/?$' --out-extension eval_prefixed -l =$(basename $d) -l 1lang=00monolingual -l 3lang=00trilingual -l weak=$(basename $tmpd) --out-dir $d $d $(find $d/00filtered/ -type d ) $(echo $d/00trilingual $d/00monolingual >/dev/null) $tmpd
rm -rf $tmpd
for lang in CMN ENG SPA
do
    d=$EDL_DIR/00filtered/$lang
    ./scripts/merge_evaluations.py --label-re='[^/]+(?=/[^/]+/[^/]+/?$)' --out-extension eval_prefixed -l =$(basename $EDL_DIR) --out-dir $d $d
done
###for f in $(find $EDL_DIR -name '*.evaluation_bydoc')
###do
###	o="${f::$((${#f} - 12))}"_prefixed
###	cat $f | grep ';docid=<micro>$' >> $o
###done


if echo $ALL_EDL | grep -q '/[0-9][0-9]*\.evaluation'
then
	TAC_GROUPRE="--group-re=(?<=/)[^/]*(?=\.)"
elif echo $ALL_EDL | grep -q '[0-9][0-9]\.evaluation'
then
	TAC_GROUPRE="--group-re=(?<=/)[^/]*(?=[0-9][0-9]\.)"
else
	TAC_GROUPRE="--group-re=(?<=/)[^/]*(?=[0-9]\.)"
fi

LABELMAP='--label-map={"fscore": "$F_1$", "precision": "$P$", "recall": "$R$", "b_cubed_plus": "B-Cubed+", "b_cubed": "B-Cubed", "strong_mention_match": "NER", "weak/strong_mention_match": "NER-weak", "strong_typed_mention_match": "NERC", "strong_all_match": "NERL", "mention_ceaf": "CEAFm", "weak/mention_ceaf": "CEAFm-weak", "entity_ceaf": "CEAFe", "entity_match": "KBIDs", "strong_all_match/strong_mention_match": "NERL/NER", "strong_typed_mention_match/strong_mention_match": "NERC/NER", "strong_nil_match": "NEN", "strong_typed_nil_match": "NENC", "strong_link_match": "NEL", "strong_typed_link_match": "NELC", "strong_typed_all_match": "NERCL", "weak/typed_mention_ceaf": "CEAFmC-weak", "typed_mention_ceaf": "CEAFmC", "mention_ceaf_plus": "CEAFm+", "typed_mention_ceaf_plus": "CEAFmC+", "NOM/strong_mention_match": "NERC for PER/NOM", "NOM/strong_all_match": "NERLC for PER/NOM", "1lang/strong_typed_nil_match": "NENC for 1-lang ents", "1lang/strong_typed_link_match": "NELC for 1-lang ents", "3lang/strong_typed_nil_match": "NENC for 3-lang ents", "3lang/strong_typed_link_match": "NELC for 3-lang ents", "mention_ceaf:is_first:span": "CEAFm-1st", "mention_ceaf;docid=<micro>": "CEAFm-doc"}'
# TODO: add styles for <micro> and is_first
STYLEMAP='--style-map={"strong_mention_match":"#4575b4/o","weak/strong_mention_match":"#4575b4/o/{\"fillstyle\":\"none\"}","strong_typed_mention_match":"#762a83/h","strong_typed_nil_match":"#fc8d59/<","strong_typed_link_match":"#91bfdb/>","strong_all_match":"black/s","entity_match":"#7fbf7b/o","mention_ceaf":"#b2182b/<","weak/typed_mention_ceaf":"#c25aa3/p/{\"fillstyle\":\"none\"}","weak/mention_ceaf":"#b2182b/</{\"fillstyle\":\"none\"}","b_cubed":"#ef8a62/>","b_cubed_plus":"#fc8d59/d","strong_all_match/strong_mention_match": "#8c510a/^", "strong_typed_all_match": "#a65aa3/p", "typed_mention_ceaf": "#c25aa3/p", "typed_mention_ceaf_plus": "#c25aa3/D", "1lang/strong_typed_nil_match":"#fc8d59/|", "1lang/strong_typed_link_match": "#91bfdb/|", "3lang/strong_typed_nil_match":"#fc8d59/1", "3lang/strong_typed_link_match": "#91bfdb/1"}'
TOP_EDL=$(get_top 6 typed_mention_ceaf $ALL_EDL)
FILTERED_TOP_EDL=$(echo $TOP_EDL | ep_ext)  # now like .../EDL//filtered/lcc20142.evaluation_prefixed
FILTERS="NW DF FAC GPE LOC ORG PER NAM NOM"

filter_measures() {
    base_measure=$1
    lang=$2
    if [ -n "$lang" ]
    then
        lang=${lang}_
    fi
    for filter in $FILTERS
    do
###        if [ "$filter" == NOM ]
###        then
###            if [ "$lang" != ENG_ -a -n "$lang" ]
###            then
###                continue
###            fi
###        fi
        echo -n -m ${lang}${filter}/$base_measure' '
    done
}

plot_edl() {
    prefix=$1
    shift
    systems=$(echo $@ | sed s/\.evaluation/.eval_prefixed/g)
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-{}.pdf $systems --lines -m typed_mention_ceaf -m mention_ceaf -m typed_mention_ceaf_plus --prf --figsize 8,4 --limits .2,.95
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-{}-ci90.pdf $(echo $systems | sed s/\.eval_prefixed/.confidence/g) --input-type confidence --ci 90 --lines -m strong_all_match -m strong_typed_all_match --prf --figsize 8,4 --limits .2,.95
    	# -m weak/strong_mention_match  removed from following
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-detection.pdf --single-plot $systems --lines --legend-ncol 2 --figsize 8,4 --limits 0.3,.95 -m strong_mention_match -m strong_typed_mention_match -m strong_all_match -m strong_typed_all_match -m entity_match
    	#-m weak/mention_ceaf  removed from following
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-clustering.pdf --single-plot $systems --lines  --figsize 8,4 --limits 0.3,.8 -m strong_all_match -m mention_ceaf -m 'mention_ceaf;docid=<micro>' -m mention_ceaf:is_first:span -m typed_mention_ceaf -m typed_mention_ceaf_plus
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-nil.pdf --single-plot $systems --lines  --figsize 8,4 --limits 0.2,.95 -m strong_all_match -m strong_typed_all_match -m strong_typed_link_match -m strong_typed_nil_match -m entity_match
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-doc.pdf --single-plot $systems --lines -m mention_ceaf -m 'mention_ceaf;docid=<micro>' -m mention_ceaf:is_first:span --figsize 8,4 --limits 0,1
    for m in typed_mention_ceaf strong_typed_all_match
    do
        ./nel plot-systems "$LABELMAP" "$STYLEMAP" --heatmap $(echo $systems | ep_ext | sed 's|00filtered/.../||g') --cmap viridis $(filter_measures $m $prefix) --figsize 4,6 --run-code "$(cat /tmp/run_code)" --run-code 'fig.savefig("'$PLOTS_DIR'/edl'$prefix'-filtered-'$m'.pdf")'
    done
	./nel plot-systems "$LABELMAP" "$STYLEMAP" --prf -o $PLOTS_DIR/edl$prefix-pernom-detection.pdf $(get_top 8 NOM/strong_mention_match $(echo $ALL_EDL| ep_ext)) --lines  --figsize 8,4 --limits 0.0,.95 -m NOM/strong_mention_match
	./nel plot-systems "$LABELMAP" "$STYLEMAP" --single-plot -o $PLOTS_DIR/edl$prefix-pernom-link.pdf $(get_top 8 NOM/strong_mention_match $(echo $ALL_EDL| ep_ext)) --lines  --figsize 8,4 --limits 0.0,.95 -m NOM/strong_mention_match -m NOM/strong_all_match

}

plot_el() {
    prefix=$1
    shift
    systems=$@
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/el$prefix-selected.pdf --single-plot $systems --lines -m mention_ceaf -m strong_all_match -m entity_match --figsize 8,4 --limits 0,1
    for m in typed_mention_ceaf strong_typed_all_match
    do
        ./nel plot-systems "$LABELMAP" "$STYLEMAP" --heatmap $(echo $systems | ep_ext | sed 's|00filtered/.../||g') --cmap viridis $(filter_measures $m $prefix) --figsize 4,6 --run-code "$(cat /tmp/run_code)" --run-code 'fig.savefig("'$PLOTS_DIR'/el'$prefix'-filtered-'$m'.pdf")'
    done
}

cat > /tmp/run_code <<HERE
fig = figures['heatmap']
ax = fig.get_axes()[0]
import numpy as np
for i in [1, ]: ax.plot([-.5, 19.5], np.array([.5, .5]) + i, '-', color='k')
labels = []
for label in ax.get_yticklabels():
    s = label.get_text()
    if '/' not in s:
        s = 'All'
    s = s.rsplit('/', 1)[0]
    if s[:4] in ['SPA_', 'CMN_', 'ENG_']:
        s = s[4:]
    labels.append(s)
ax.set_yticklabels(labels)
fig.tight_layout()
HERE
filter_prefixed() {
    measure=$1
    echo $measure
    for prefix in $FILTERS
    do
        echo -m $prefix/$measure
    done
}


report_table() {
	measures="$1"
	abbr="$2"
	root_dir="$3"
	shift;shift;shift
	
	opts='-p -b \\\\textbf{%s} '"$measures"
	ncols=$(($(echo $abbr | wc -w) * 3 + 1))
	echo '\newcommand{\tbsec}[1]{\multicolumn{'$ncols'}{c}{\textsc{#1}}\\\midrule}'
	echo -n '\begin{tabular}{l*{'$(echo $abbr | wc -w)'}{|*2cS{blue!10}}}'
	echo '\toprule'
	echo -n '\multirow{2}{*}{System}'
	for m in $abbr
	do
		echo -n '&\multicolumn{3}{c|}{'$m'}'
	done | sed 's/\(.*\)|/\1/'
	echo '\\'
	for m in $abbr
	do
		echo -n '&$P$&$R$&$F_1$'
	done
	echo '\\'
	_sec() {
		head=$1
		nsys=$2
		dir=$3
		echo '\midrule'
		echo \\tbsec{$head}
		systems=$(get_top $nsys typed_mention_ceaf $dir/*.evaluation | ep_ext)
		scripts/report.sh $opts $systems | tr '\t' '&'| sed 's/$/\\\\/;s/_/\\_/g'| tail -n+2
	}
	_sec Trilingual $1 $root_dir; shift
	_sec Chinese $1 $root_dir/00filtered/CMN/; shift
	_sec English $1 $root_dir/00filtered/ENG/; shift
	_sec Spanish $1 $root_dir/00filtered/SPA/; shift
	echo '\bottomrule'
	echo '\end{tabular}'
}
# missing -m weak/mention_ceaf    == \\ceafmw
report_table "-m strong_mention_match -m strong_typed_mention_match -m strong_typed_all_match -m entity_match -m mention_ceaf -m typed_mention_ceaf -m mention_ceaf;docid=<micro> -m mention_ceaf:is_first:span" \
	"\\ner \\nerc \\nerlc \\etag \\ceafm \\ceafmc \\ceafmwithin \\ceafmfirst" \
	$EDL_DIR \
	20 20 20 20 \
	> $PLOTS_DIR/edl-report.tex

plot_edl '' $TOP_EDL
plot_edl ENG $(get_top 8 typed_mention_ceaf $EDL_DIR/00filtered/ENG/*.evaluation)
plot_edl CMN $(get_top 7 typed_mention_ceaf $EDL_DIR/00filtered/CMN/*.evaluation)
plot_edl SPA $(get_top 7 typed_mention_ceaf $EDL_DIR/00filtered/SPA/*.evaluation)

for m in strong_typed_all_match strong_typed_mention_match
do
	./nel plot-systems --label-map='{"fscore":"$F_1$","'$m'":"all","CMN/'$m'":"Chinese", "ENG/'$m'":"English", "SPA/'$m'":"Spanish"}' "$STYLEMAP" -o $PLOTS_DIR/edl-languages-$m.pdf -m $m -m CMN/$m -m ENG/$m -m SPA/$m --single-plot $(echo $TOP_EDL | ep_ext) --lines  --figsize 8,4 --limits 0.0,.95
	./nel plot-systems --legend-ncol=2 --label-map='{"fscore":"$F_1$","'$m'":"all","FAC/'$m'":"FAC", "GPE/'$m'":"GPE", "LOC/'$m'":"LOC", "ORG/'$m'":"ORG", "PER/'$m'":"PER", "NAM/'$m'":"NAM", "NOM/'$m'":"NOM"}' "$STYLEMAP" -o $PLOTS_DIR/edl-types-$m.pdf -m $m -m FAC/$m -m GPE/$m -m LOC/$m -m ORG/$m -m PER/$m -m NOM/$m --single-plot $(echo $TOP_EDL | ep_ext) --lines  --figsize 8,4 --limits 0.0,.95
done
###./nel plot-systems "$LABELMAP" "$STYLEMAP" --recall-only -o $PLOTS_DIR/edl-crossling-recall.pdf --single-plot $(echo $TOP_EDL | ep_ext) --lines  --figsize 8,4 --limits 0.0,.95 --legend-ncol 3 -m strong_typed_link_match -m strong_typed_nil_match -m 1lang/strong_typed_link_match -m 1lang/strong_typed_nil_match -m 3lang/strong_typed_link_match -m 3lang/strong_typed_nil_match
