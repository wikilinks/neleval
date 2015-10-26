#!/usr/bin/env bash
#
# Generate many plots
set -e

OFFICIAL_MEASURES="-m strong_mention_match -m strong_typed_mention_match -m strong_all_match -m typed_mention_ceaf -m mention_ceaf"
DEFAULT_MEASURE=typed_mention_ceaf

# TODO: Determine figure sizes dynamically
square_figsize="6,6"
syscols_figsize="17,4"
teamcols_figsize="8,4"

SCR=`dirname $0`
. $SCR/_run_tac_plots.sh
