set -e
get_eval_prf() {
	# Extracts columns corresponding to measures from `evaluate` output
    script='
        {
            data[$8] = $5 "\t" $6 "\t" $7;
        }
        END {
            n = split(measures, a, ",");
            out = "";
            for (i = 1; i <= n; i++) {
                if (length(out))
                    out = out "\t";
                d = data[a[i]];
                if (length(d) == 0) {
                    print "Could not find measure " a[i] > "/dev/stderr";
                    exit 1;
                }
                out = out d;
            }
            print out
        }
    '
	awk -v measures="$(echo $@ | tr ' ' ,)" "$script"
}


