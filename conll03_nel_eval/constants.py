
import os

GITHUB_OWNER = 'wikilinks'
GITHUB_REPO = 'conll03_nel_eval'
GITHUB_BRANCH = 'master'
REFERENCE_OUTPUT_FMT = os.path.join('references', '{name}-{mentions}-{mapping}.testb.txt')
REFERENCE_README_FMT = os.path.join('references', '{name}.README')
