#!/bin/bash
OUTPUTS=$1
echo 'Mapping gold standard to latest API'
for f in gold-unmapped.testb.txt tagme-unmapped.testb.txt aida-gold-mentions-unmapped.testb.txt radford-unsup-unmapped.testb.txt radford-unsup-gold-mentions-unmapped.testb.txt; do
    echo $f
    ./cne prepare -m mappings/map-testb-fromapi-20140227.tsv $OUTPUTS/$f > $OUTPUTS/`echo $f | sed 's/unmapped/api20140227/'`
done;

echo 'Evaluating end-to-end'
./cne evaluate -g $OUTPUTS/gold-api20140227.testb.txt $OUTPUTS/tagme-api20140227.testb.txt > $OUTPUTS/tagme.eval
./cne evaluate -g $OUTPUTS/gold-api20140227.testb.txt $OUTPUTS/radford-unsup-api20140227.testb.txt > $OUTPUTS/radford.eval

echo 'Evaluating linkables'
./cne filter-mentions '.' --field 4 --aux $OUTPUTS/aida-gold-mentions-api20140227.testb.txt $OUTPUTS/gold-api20140227.testb.txt > $OUTPUTS/gold-api20140227.linkable.testb.txt
./cne filter-mentions '.' --field 4 --aux $OUTPUTS/aida-gold-mentions-api20140227.testb.txt $OUTPUTS/radford-unsup-gold-mentions-api20140227.testb.txt > $OUTPUTS/radford-unsup-gold-mentions-api20140227.linkable.testb.txt
./cne evaluate -g $OUTPUTS/gold-api20140227.linkable.testb.txt $OUTPUTS/aida-gold-mentions-api20140227.testb.txt > $OUTPUTS/aida.linkable.eval
./cne evaluate -g $OUTPUTS/gold-api20140227.testb.txt $OUTPUTS/radford-unsup-gold-mentions-api20140227.linkable.testb.txt > $OUTPUTS/radford.linkable.eval

echo 'Comparing errors'
./cne analyze -s -g $OUTPUTS/gold-api20140227.testb.txt $OUTPUTS/tagme-api20140227.testb.txt > $OUTPUTS/tagme.analysis
./cne analyze -s -g $OUTPUTS/gold-api20140227.testb.txt $OUTPUTS/radford-unsup-api20140227.testb.txt > $OUTPUTS/radford.analysis
./cne analyze -s -g $OUTPUTS/gold-api20140227.linkable.testb.txt $OUTPUTS/aida-gold-mentions-api20140227.testb.txt > $OUTPUTS/aida.linkable.analysis
./cne analyze -s -g $OUTPUTS/gold-api20140227.testb.txt $OUTPUTS/radford-unsup-gold-mentions-api20140227.linkable.testb.txt > $OUTPUTS/radford.linkable.analysis
