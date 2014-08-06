
# Configuration constants
ALL_CMATCHES = 'all'
MUC_CMATCHES = 'muc'
LUO_CMATCHES = 'luo'
CAI_STRUBE_CMATCHES = 'cai'
TAC_CMATCHES = 'tac'
TMP_CMATCHES = 'tmp'
NO_CMATCHES = 'none'
CMATCH_SETS = {
    ALL_CMATCHES: [
        'mention_ceaf',
        'entity_ceaf',
        'b_cubed',
        'pairwise',
        'muc',
        #'mention_cs_ceaf',
        #'entity_cs_ceaf',
        #'cs_b_cubed',
        'b_cubed_plus',
        ],
    MUC_CMATCHES: [
        'muc',
        ],
    LUO_CMATCHES: [
        'muc',
        'b_cubed',
        'mention_ceaf',
        'entity_ceaf',
        ],
    CAI_STRUBE_CMATCHES: [
        'cs_b_cubed',
        'mention_cs_ceaf',
    ],
    TAC_CMATCHES: [
        'mention_ceaf',
        'b_cubed',
        ],
    TMP_CMATCHES: [
        'mention_ceaf',
        'entity_ceaf',
        'pairwise',
        ],
}
DEFAULT_CMATCH_SET = ALL_CMATCHES


ALL_LMATCHES = 'all'
CORNOLTI_WWW13_LMATCHES = 'cornolti'
HACHEY_ACL14_LMATCHES = 'hachey'
TAC_LMATCHES = 'tac'
TAC14_LMATCHES = 'tac14'

LMATCH_SETS = {
    ALL_LMATCHES: [
        'strong_mention_match',
        'strong_linked_mention_match',
        'strong_link_match',
        'strong_nil_match',
        'strong_all_match',
        'strong_typed_all_match',
        'entity_match',
        ],
    CORNOLTI_WWW13_LMATCHES: [
        'strong_linked_mention_match',
        'strong_link_match',
        'entity_match',
        ],
    HACHEY_ACL14_LMATCHES: [
        'strong_mention_match', # full ner
        'strong_linked_mention_match',
        'strong_link_match',
        'entity_match',
        ],
    TAC_LMATCHES: [
        'strong_link_match', # recall equivalent to kb accuracy before 2014
        'strong_nil_match', # recall equivalent to nil accuracy before 2014
        'strong_all_match', # equivalent to overall accuracy before 2014
        'strong_typed_all_match',  # wikification f-score for TAC 2014
        ],
    TAC14_LMATCHES: [
        'strong_typed_all_match', # wikification f-score for TAC 2014
        ]
    }
DEFAULT_LMATCH_SET = HACHEY_ACL14_LMATCHES

