#!/bin/bash
echo "Testing eval against repository version"
for new in `find outputs -name '*eval'`; do
    org=references/`basename $new`
    echo "  diffing $org with $new"
    diff $org $new
done
