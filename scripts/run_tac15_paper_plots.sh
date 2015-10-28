#!/bin/bash
EDL_DIR=$1
EL_DIR=$2
PLOTS_DIR=$3
if [ -z "$PLOTS_DIR" ]
then
    echo Usage: EDL_DIR EL_DIR' PLOTS_DIR'
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
    systems=$@
    ./nel rank-systems $GROUPRE -m mention_ceaf --max $n --group-limit 1 $systems | tail -n+2 | cut -f7
}

LABELMAP='--label-map={"fscore": "$F_1$", "precision": "$P$", "recall": "$R$", "b_cubed_plus": "B-Cubed+", "b_cubed": "B-Cubed", "strong_mention_match": "NER", "strong_typed_mention_match": "NERC", "strong_all_match": "NERL", "mention_ceaf": "CEAFm", "entity_ceaf": "CEAFe", "entity_match": "KBIDs", "strong_all_match/strong_mention_match": "NERL/NER", "strong_typed_mention_match/strong_mention_match": "NERC/NER", "strong_nil_match": "NEN", "strong_link_match": "NEL", "strong_typed_all_match": "NERCL", "typed_mention_ceaf": "Typed CEAFm", "mention_ceaf_plus": "CEAFm+", "typed_mention_ceaf_plus": "Typed CEAFm+"}'
GROUPRE="--group-re=(?<=/)[^/]*(?=[0-9]\.)"
STYLEMAP='--style-map={"strong_mention_match":"#4575b4/o","strong_typed_mention_match":"#762a83/h","strong_nil_match":"#fc8d59/<","strong_link_match":"#91bfdb/>","strong_all_match":"black/s","entity_match":"#7fbf7b/o","mention_ceaf":"#b2182b/<","b_cubed":"#ef8a62/>","b_cubed_plus":"#fc8d59/d","strong_all_match/strong_mention_match": "#8c510a/^", "strong_typed_all_match": "#a65aa3/p", "typed_mention_ceaf": "#c25aa3/p", "typed_mention_ceaf_plus": "#c25aa3/D"}'
ALL_EDL=$EDL_DIR/*.evaluation
TOP_EDL=$(get_top 6 $ALL_EDL)
FILTERED_TOP_EDL=$(echo $TOP_EDL | ep_ext)  # now like .../EDL//filtered/lcc20142.evaluation_prefixed
ALL_EL=$EL_DIR/*.evaluation
TOP_EL=$(get_top 6 $ALL_EL)
FILTERED_TOP_EL=$(echo $TOP_EL | ep_ext)  # now like .../EL//filtered/lcc20142.evaluation_prefixed
FILTERS=$(ls -d $EDL_DIR/00filtered/* | sed 's|.*/\([^/]*\)$|\1|g')

for f in $EDL_DIR/*.evaluation $EL_DIR/*.evaluation
do
    pushd $(dirname "$f") >/dev/null
    awk '
    BEGIN{OFS="\t"}
    $7 == "fscore" && NR == 1 {print}
    $7 != "fscore" {
        m= match(FILENAME,/\/[^\/]*\/[^\/]*$/);
        pref=substr(FILENAME, m);
        if (m <= 0) {
            pref="";
        } else {
            pref=substr(pref, 2, match(substr(pref, 2), /\//) );
        };
        $8 = pref $8;
        print}
    ' $(find . -name $(basename "$f")) > $(echo $(basename "$f") | ep_ext)
    popd >/dev/null
done

plot_edl() {
    prefix=$1
    shift
    systems=$@
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-{}.pdf $systems --lines -m mention_ceaf --prf --figsize 8,4 --limits .2,.9
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-{}.new.pdf $(echo $systems | sed s/\.evaluation/.confidence/g) --input-type confidence --ci 90 --lines -m strong_all_match --prf --figsize 8,4 --limits .2,.9
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-detection.pdf --single-plot $systems --lines  --figsize 8,4 --limits 0.2,.9 -m strong_mention_match -m strong_typed_mention_match -m strong_all_match -m strong_typed_all_match -m entity_match -m strong_all_match/strong_mention_match
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-clustering.pdf --single-plot $systems --lines  --figsize 8,4 --limits 0.2,.9 -m strong_all_match -m mention_ceaf -m typed_mention_ceaf -m typed_mention_ceaf_plus -m b_cubed -m b_cubed_plus
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/edl$prefix-nil.pdf --single-plot $systems --lines  --figsize 8,4 --limits 0.2,.9 -m strong_all_match -m strong_link_match -m strong_nil_match -m entity_match
}

plot_el() {
    prefix=$1
    shift
    systems=$@
    ./nel plot-systems "$LABELMAP" "$STYLEMAP" -o $PLOTS_DIR/el$prefix-selected.pdf --single-plot $systems --lines -m b_cubed_plus -m b_cubed -m mention_ceaf -m strong_all_match -m entity_match --figsize 8,4 --limits 0,1
}

####./nel compose-measures -r strong_all_match strong_mention_match -r strong_typed_mention_match strong_mention_match $ALL_EDL
plot_edl '' $TOP_EDL
plot_el '' $TOP_EL
plot_edl ENG $(get_top 8 $EDL_DIR/00filtered/ENG/*.evaluation)
plot_el ENG $(get_top 9 $EL_DIR/00filtered/ENG/*.evaluation)
plot_edl CMN $(get_top 5 $EDL_DIR/00filtered/CMN/*.evaluation)
plot_el CMN $(get_top 5 $EL_DIR/00filtered/CMN/*.evaluation)
plot_edl SPA $(get_top 7 $EDL_DIR/00filtered/SPA/*.evaluation)
plot_el SPA $(get_top 4 $EL_DIR/00filtered/SPA/*.evaluation)

cat > /tmp/run_code <<HERE
fig = figures['heatmap']
ax = fig.get_axes()[0]
import numpy as np
for i in [0, 3, 6, 9, 12]: ax.plot([-.5, 19.5], np.array([.5, .5]) + i, '-', color='k')
labels = []
for label in ax.get_yticklabels():
    s = label.get_text()
    if '/' not in s:
        s = 'All'
    s = s.rsplit('/', 1)[0]
    s = s.replace('_', ' & ')
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
./nel plot-systems "$LABELMAP" "$STYLEMAP" --heatmap $FILTERED_TOP_EDL --cmap Reds_r $(for prefix in $FILTERS; do echo -m ${prefix}/mention_ceaf; done) --figsize 6,6 --run-code "$(cat /tmp/run_code)" --run-code 'fig.savefig("'$PLOTS_DIR'/edl-filtered-'$prefix'-mention_ceaf.pdf")'
./nel plot-systems "$LABELMAP" "$STYLEMAP" --heatmap $FILTERED_TOP_EDL --cmap Reds_r $(for prefix in $FILTERS; do echo -m ${prefix}/strong_all_match; done) --figsize 6,6  --run-code "$(cat /tmp/run_code)" --run-code 'fig.savefig("'$PLOTS_DIR'/edl-filtered-'$prefix'-strong_all_match.pdf")'
./nel plot-systems "$LABELMAP" "$STYLEMAP" --heatmap $FILTERED_TOP_EL --cmap Reds_r $(for prefix in $FILTERS; do echo -m ${prefix}/strong_all_match; done) --figsize 6,6 --run-code "$(cat /tmp/run_code)" --run-code 'fig.savefig("'$PLOTS_DIR'/el-filtered-'$prefix'-strong_all_match.pdf")'
