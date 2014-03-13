#!/bin/bash
set -e
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/AIDA-YAGO2-dataset.tsv"
    exit 1
fi
if [ ! -f $gold ]; then
    echo "Usage: $0 /path/to/AIDA-YAGO2-dataset.tsv"
    exit 1
fi

mkdir -p outputs
aidadata=$1
mappingid='api20140227'
mapping='mappings/map-testb-fromapi-20140227.tsv'
gold='gold-unmapped.testb.txt'

echo 'Filtering testb from gold standard'
./cne prepare -k '.*testb.*' $aidadata > outputs/$gold

echo 'Stitching releases'
for f in tagme_336-unmapped.testb.txt tagme_289-unmapped.testb.txt aida-gold-mentions-unmapped.testb.txt schwa-unsup-unmapped.testb.txt schwa-unsup-gold-mentions-unmapped.testb.txt; do
    ./cne stitch -g outputs/$gold references/$f > outputs/$f
done

echo "Mapping gold standard to latest API $mappingid"
for f in $gold tagme_289-unmapped.testb.txt tagme_336-unmapped.testb.txt aida-gold-mentions-unmapped.testb.txt schwa-unsup-unmapped.testb.txt schwa-unsup-gold-mentions-unmapped.testb.txt; do
    ./cne prepare -m $mapping outputs/$f > outputs/`echo $f | sed "s/unmapped/$mappingid/"`
done;
mapped_gold=outputs/`echo $gold | sed "s/unmapped/$mappingid/"`

echo 'Evaluating end-to-end'
./cne evaluate -g $mapped_gold outputs/tagme_289-$mappingid.testb.txt > outputs/tagme_289.eval
./cne evaluate -g $mapped_gold outputs/tagme_336-$mappingid.testb.txt > outputs/tagme_336.eval
./cne evaluate -g $mapped_gold outputs/schwa-unsup-$mappingid.testb.txt > outputs/schwa.eval

echo 'Filtering linkables'
./cne filter-mentions '.' --field 4 --aux outputs/aida-gold-mentions-$mappingid.testb.txt $mapped_gold > outputs/gold-$mappingid.linkable.testb.txt
./cne filter-mentions '.' --field 4 --aux outputs/aida-gold-mentions-$mappingid.testb.txt outputs/schwa-unsup-gold-mentions-$mappingid.testb.txt > outputs/schwa-unsup-gold-mentions-$mappingid.linkable.testb.txt

echo 'Evaluating linkables'
./cne evaluate -g outputs/gold-$mappingid.linkable.testb.txt outputs/aida-gold-mentions-$mappingid.testb.txt > outputs/aida.linkable.eval
./cne evaluate -g outputs/gold-$mappingid.linkable.testb.txt outputs/schwa-unsup-gold-mentions-$mappingid.linkable.testb.txt > outputs/schwa.linkable.eval

echo 'Running analysis'
./cne analyze -s -g $mapped_gold outputs/tagme_289-$mappingid.testb.txt > outputs/tagme_289.analysis
./cne analyze -s -g $mapped_gold outputs/tagme_336-$mappingid.testb.txt > outputs/tagme_336.analysis
./cne analyze -s -g $mapped_gold outputs/schwa-unsup-$mappingid.testb.txt > outputs/schwa.analysis
./cne analyze -s -g outputs/gold-$mappingid.linkable.testb.txt outputs/aida-gold-mentions-$mappingid.testb.txt > outputs/aida.linkable.analysis
./cne analyze -s -g outputs/gold-$mappingid.linkable.testb.txt outputs/schwa-unsup-gold-mentions-$mappingid.linkable.testb.txt > outputs/schwa.linkable.analysis

echo 'Complete: see outputs/*eval and outputs/*analysis'
