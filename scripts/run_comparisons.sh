#!/bin/bash
echo 'Mapping gold standard to latest API'
for f in gold-unmapped.testb.txt tagme-unmapped.testb.txt aida-gold-mentions-unmapped.testb.txt radford-unsup-unmapped.testb.txt radford-unsup-gold-mentions-unmapped.testb.txt; do
    echo $f
    ./cne prepare -m mappings/map-testb-fromapi-20140227.tsv outputs/$f > outputs/`echo $f | sed 's/unmapped/api20140227/'`
done;

echo 'Evaluating end-to-end'
./cne evaluate -g outputs/gold-api20140227.testb.txt outputs/tagme-api20140227.testb.txt > outputs/tagme.eval
./cne evaluate -g outputs/gold-api20140227.testb.txt outputs/radford-unsup-api20140227.testb.txt > outputs/radford.eval

echo 'Evaluating linkables'
./cne filter-mentions '.' --field 4 --aux outputs/aida-gold-mentions-api20140227.testb.txt outputs/gold-api20140227.testb.txt > outputs/gold-api20140227.linkable.testb.txt
./cne filter-mentions '.' --field 4 --aux outputs/aida-gold-mentions-api20140227.testb.txt outputs/radford-unsup-gold-mentions-api20140227.testb.txt > outputs/radford-unsup-gold-mentions-api20140227.linkable.testb.txt
./cne evaluate -g outputs/gold-api20140227.linkable.testb.txt outputs/aida-gold-mentions-api20140227.testb.txt > outputs/aida.linkable.eval
./cne evaluate -g outputs/gold-api20140227.linkable.testb.txt outputs/radford-unsup-gold-mentions-api20140227.linkable.testb.txt > outputs/radford.linkable.eval

echo 'Comparing errors'
./cne analyze -s -g outputs/gold-api20140227.testb.txt outputs/tagme-api20140227.testb.txt > outputs/tagme.analysis
./cne analyze -s -g outputs/gold-api20140227.testb.txt outputs/radford-unsup-api20140227.testb.txt > outputs/radford.analysis
./cne analyze -s -g outputs/gold-api20140227.linkable.testb.txt outputs/aida-gold-mentions-api20140227.testb.txt > outputs/aida.linkable.analysis
./cne analyze -s -g outputs/gold-api20140227.linkable.testb.txt outputs/radford-unsup-gold-mentions-api20140227.linkable.testb.txt > outputs/radford.linkable.analysis
