#!/bin/bash

set -e

echo "Checking all commands have documentation pages"

diff <(
	neleval --help |
	grep '^ *{' |
	head -n1 |
	sed 's/^[ {]*//;s/[} ]*$//' |
	tr "," "\n" |
	sort
	) <(
	ls -1 doc/commands/ |
		grep -v main.rst |
		sed 's|.*/||;s/.rst$//' |
		sort
	)
