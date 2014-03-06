#!/bin/bash
set -e
gold=$1
if [ ! -f $gold ]; then
    echo 'Could not find gold!'
    exit 1
fi
mkdir -p outputs
echo 'Stitching releases'
cp $gold outputs
for f in tagme-unmapped.testb.txt aida-gold-mentions-unmapped.testb.txt schwa-unsup-unmapped.testb.txt schwa-unsup-gold-mentions-unmapped.testb.txt; do
    ./cne stitch -g $gold references/$f > outputs/$f
done

echo 'Mapping gold standard to latest API'
for f in $gold tagme-unmapped.testb.txt aida-gold-mentions-unmapped.testb.txt schwa-unsup-unmapped.testb.txt schwa-unsup-gold-mentions-unmapped.testb.txt; do
    ./cne prepare -m mappings/map-testb-fromapi-20140227.tsv outputs/$f > outputs/`echo $f | sed 's/unmapped/api20140227/'`
done;
mapped_gold=outputs/`echo $gold | sed 's/unmapped/api20140227/'`

echo 'Evaluating end-to-end'
./cne evaluate -g $mapped_gold outputs/tagme-api20140227.testb.txt > outputs/tagme.eval
./cne evaluate -g $mapped_gold outputs/schwa-unsup-api20140227.testb.txt > outputs/schwa.eval

echo 'Evaluating linkables'
./cne filter-mentions '.' --field 4 --aux outputs/aida-gold-mentions-api20140227.testb.txt $mapped_gold > outputs/gold-api20140227.linkable.testb.txt
./cne filter-mentions '.' --field 4 --aux outputs/aida-gold-mentions-api20140227.testb.txt outputs/schwa-unsup-gold-mentions-api20140227.testb.txt > outputs/schwa-unsup-gold-mentions-api20140227.linkable.testb.txt
./cne evaluate -g outputs/gold-api20140227.linkable.testb.txt outputs/aida-gold-mentions-api20140227.testb.txt > outputs/aida.linkable.eval
./cne evaluate -g outputs/gold-api20140227.linkable.testb.txt outputs/schwa-unsup-gold-mentions-api20140227.linkable.testb.txt > outputs/schwa.linkable.eval

echo 'Comparing errors'
./cne analyze -s -g $mapped_gold outputs/tagme-api20140227.testb.txt > outputs/tagme.analysis
./cne analyze -s -g $mapped_gold outputs/schwa-unsup-api20140227.testb.txt > outputs/schwa.analysis
./cne analyze -s -g outputs/gold-api20140227.linkable.testb.txt outputs/aida-gold-mentions-api20140227.testb.txt > outputs/aida.linkable.analysis
./cne analyze -s -g outputs/gold-api20140227.linkable.testb.txt outputs/schwa-unsup-gold-mentions-api20140227.linkable.testb.txt > outputs/schwa.linkable.analysis
