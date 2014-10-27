set -e
SCR=`dirname $0`

get_eval_prf() {
	# Extracts columns corresponding to a measure from `evaluate` output
	awk '{if ($8 == "'$1'") print}' \
        | cut -f5,6,7 \
        | tr '\n' '\t'
}
