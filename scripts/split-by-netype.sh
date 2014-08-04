#!/bin/bash

usage() {
	echo 'Select mentions in linked text by CoNLL03 Shared Task entity types.

Usage: '$0' conll03ner/etc/tags.eng.testb linked1 ...

Produces files such as linked1.PER, linked1.LOC, linked1.ORG, linked1.MISC
containing only the corresponding entity mentions in the evaluation format.
' >&2
	exit 2
}

CNE=./nel

CONLL_TAGS_PATH=$1
shift

if [ -z "$CONLL_TAGS_PATH" ]
then
	usage
fi

while [ -n "$1" ]
do
	for netype in PER LOC ORG MISC
	do
		echo $1.$netype >&2
		$CNE filter-mentions $netype -f3 -d' ' --aux "$CONLL_TAGS_PATH" "$1" > "$1.$netype" || exit 1
	done

	shift
done
