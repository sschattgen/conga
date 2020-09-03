import sys
import random
from .basic import *
from . import translation
from .all_genes import all_genes, gap_character
from .genetic_code import genetic_code, reverse_genetic_code
from . import logo_tools

all_trbd_nucseq = {}
for organism in all_genes:
    d_ids = sorted( ( id for id,g in all_genes[organism].items() if g.region == 'D' and g.chain == 'B' ) )
    all_trbd_nucseq[ organism ] = dict( ( ( d_ids.index(id)+1, all_genes[organism][id].nucseq ) for id in d_ids ) )

########################################################################################################################
default_mismatch_score_for_cdr3_nucseq_probabilities = -4 ## blast is -3
default_mismatch_score_for_junction_analysis = -4 ## blast is -3

def count_matches( a,b,mismatch_score=-3 ):
    assert a[0].lower() == a[0]
    #assert a[0].upper() == a[0]
    ## from the beginning
    match_score    = 1
    best_score=0
    score=0
    num_matches = 0
    for i in range(min(len(a),len(b))):
        if a[i] == b[i] or logo_tools.nuc_match_lower_case.get( (a[i],b[i]), False ):
            score += match_score
        else:
            score += mismatch_score
        if score >= best_score: ## gt OR EQUAL! take longer matched regions
            best_score = score
            num_matches = i+1
    return num_matches



def get_v_cdr3_nucseq( organism, v_gene, paranoid = False ):
    vg = all_genes[organism][v_gene]
    ab = vg.chain

    v_nucseq  = vg.nucseq
    v_nucseq_offset = vg.nucseq_offset
    v_nucseq = v_nucseq[ v_nucseq_offset: ]

    v_alseq = vg.alseq
    alseq_cpos = vg.cdr_columns[-1][0] -1
    #print organism, v_gene, old_v_alseq
    #print organism, v_gene, v_alseq
    numgaps = v_alseq[:alseq_cpos].count('.')
    v_cpos = alseq_cpos - numgaps
    v_nucseq = v_nucseq[3*v_cpos:] ## now v_nucseq starts with the 'C' codon


    # the following hacky adjustment is now incorporated in the dbfile
    # if organism == 'mouse':
    #     if v_gene == 'TRAV13D-1*01':
    #         #-----------------------------------
    #         #../../../tmp.blat:mismatch: V 6 imgt: a genome: t TRAV13D-1*01
    #         #tmp.whoah:whoah  6 act: t  98.7 exp: a   1.1 TRAV13D-1*01 TRAV13-1*01 620
    #         #tmp.whoah:whoah: expected: caaggtatcgtgt consensus: caaggtttcgtgt TRAV13D-1*01 620
    #         #tmp.3.whoah:whoah  6 act: t  97.4 exp: a   1.4 TRAV13D-1*01 TRAV13-1*01 642
    #         #tmp.3.whoah:whoah: expected: caaggtatcgtgt consensus: caaggtttcgtgt TRAV13D-1*01 642
    #         #tmp.la_mc.whoah:whoah  6 act: t  89.0 exp: a   7.0 TRAV13D-1*01 TRAV13-1*01 100
    #         #tmp.la_mc.whoah:whoah: expected: caaggtatcgtgt consensus: caaggtttcgtgt TRAV13D-1*01 100
    #         assert v_nucseq == 'tgtgctatggaac' ## CAM ## THIS WILL FAIL SINCE WE ADDED THIS TO THE DB...
    #         v_nucseq         = 'tgtgctttggaac' ## CAL


    return v_nucseq


def get_j_cdr3_nucseq( organism, j_gene, paranoid = False ):
    jg = all_genes[organism][j_gene]
    ab = jg.chain

    j_nucseq  = jg.nucseq
    j_nucseq_offset = jg.nucseq_offset

    ## goes up to (but not including) GXG
    num_genome_j_aas_in_loop = len( jg.cdrs[0].replace(gap_character,''))

    ## trim j_nucseq so that it extends up to the F/W position
    j_nucseq = j_nucseq[:3*num_genome_j_aas_in_loop + j_nucseq_offset]


    # the following hacky adjustments are now incorporated in the dbfile
    # if organism == 'mouse':
    #     if j_gene == 'TRAJ47*01':
    #         # -----------------------------------
    #         # ../../../tmp.blat:mismatch: J 2 imgt: c genome: g TRAJ47*01
    #         # ../../../tmp.blat:mismatch: J 24 imgt: g genome: t TRAJ47*01
    #         # tmp.whoah:whoah  2 act: g  81.9 exp: c   4.7 TRAJ47*01 TRAJ47*01 1412
    #         # tmp.whoah:whoah 24 act: t  82.7 exp: g  16.8 TRAJ47*01 TRAJ47*01 1412
    #         # tmp.whoah:whoah: expected: tgcactatgcaaacaagatgatctgt consensus: tggactatgcaaacaagatgatcttt TRAJ47*01 1412
    #         # tmp.3.whoah:whoah  2 act: g  81.6 exp: c   5.0 TRAJ47*01 TRAJ47*01 1362
    #         # tmp.3.whoah:whoah 24 act: t  82.7 exp: g  16.6 TRAJ47*01 TRAJ47*01 1362
    #         # tmp.3.whoah:whoah: expected: tgcactatgcaaacaagatgatctgt consensus: tggactatgcaaacaagatgatcttt TRAJ47*01 1362
    #         # tmp.la_mc.whoah:whoah  2 act: g  79.6 exp: c   5.3 TRAJ47*01 TRAJ47*01 113
    #         # tmp.la_mc.whoah:whoah 24 act: t  99.1 exp: g   0.9 TRAJ47*01 TRAJ47*01 113
    #         # tmp.la_mc.whoah:whoah: expected: tgcactatgcaaacaagatgatctgt consensus: tggactatgcaaacaagatgatcttt TRAJ47*01 113
    #         assert j_nucseq == 'tgcactatgcaaacaagatgatctgt' ## C at end
    #         j_nucseq         = 'tggactatgcaaacaagatgatcttt' ## F at end
    #     elif j_gene == 'TRAJ24*01':
    #         # -----------------------------------
    #         # ../../../tmp.blat:unaligned: J 0 TRAJ24*01
    #         # ../../../tmp.blat:unaligned: J 1 TRAJ24*01
    #         # ../../../tmp.blat:gapped: J 6 TRAJ24*01
    #         # tmp.whoah:whoah  2 act: c  60.3 exp: a  15.3 TRAJ24*01 TRAJ24*01 464
    #         # tmp.whoah:whoah  4 act: a  88.6 exp: c   2.8 TRAJ24*01 TRAJ24*01 464
    #         # tmp.whoah:whoah  5 act: c  93.3 exp: t   1.5 TRAJ24*01 TRAJ24*01 464
    #         # tmp.whoah:whoah  6 act: t  97.2 exp: g   1.1 TRAJ24*01 TRAJ24*01 464
    #         # tmp.whoah:whoah: expected: tgaactggccagtttggggaaactgcagttt consensus: gacaactgccagtttggggaaactgcagttt TRAJ24*01 464
    #         # tmp.3.whoah:whoah  2 act: c  60.8 exp: a  13.9 TRAJ24*01 TRAJ24*01 475
    #         # tmp.3.whoah:whoah  4 act: a  86.3 exp: c   4.2 TRAJ24*01 TRAJ24*01 475
    #         # tmp.3.whoah:whoah  5 act: c  94.5 exp: t   1.1 TRAJ24*01 TRAJ24*01 475
    #         # tmp.3.whoah:whoah  6 act: t  98.1 exp: g   1.1 TRAJ24*01 TRAJ24*01 475
    #         # tmp.3.whoah:whoah: expected: tgaactggccagtttggggaaactgcagttt consensus: gacaactgccagtttggggaaactgcagttt TRAJ24*01 475
    #         # tmp.la_mc.whoah:whoah  2 act: c  75.3 exp: a   4.3 TRAJ24*01 TRAJ24*01 93
    #         # tmp.la_mc.whoah:whoah  4 act: a  89.2 exp: c   2.2 TRAJ24*01 TRAJ24*01 93
    #         # tmp.la_mc.whoah:whoah  5 act: c  97.8 exp: t   1.1 TRAJ24*01 TRAJ24*01 93
    #         # tmp.la_mc.whoah:whoah  6 act: t  98.9 exp: g   0.0 TRAJ24*01 TRAJ24*01 93
    #         # tmp.la_mc.whoah:whoah: expected: tgaactggccagtttggggaaactgcagttt consensus: gacaactgccagtttggggaaactgcagttt TRAJ24*01 93
    #         assert j_nucseq == 'tgaactggccagtttggggaaactgcagttt'
    #         j_nucseq         = 'gacaactgccagtttggggaaactgcagttt'
    #         ## take the consensus
    #         ## given that there's an indel (and the alignment to the genome starts at j sequence position 3)
    #         ## it's hard to tell what to do at the beginning...



    return j_nucseq




def analyze_junction( organism, v_gene, j_gene, cdr3_protseq, cdr3_nucseq, force_d_id=0, return_cdr3_nucseq_src=False ):
    #print organism, v_gene, j_gene, cdr3_protseq, cdr3_nucseq
    #assert v_gene.startswith('TR') #and v_gene[2] == j_gene[2]
    ab = all_genes[organism][v_gene].chain
    v_nucseq = get_v_cdr3_nucseq( organism, v_gene )
    j_nucseq = get_j_cdr3_nucseq( organism, j_gene )
    ## how far out do we match
    num_matched_v = count_matches( v_nucseq, cdr3_nucseq, default_mismatch_score_for_junction_analysis )

    num_matched_j = count_matches( ''.join( reversed( list( j_nucseq ) )),
                                   ''.join( reversed( list( cdr3_nucseq ))),
                                   default_mismatch_score_for_junction_analysis )


    if num_matched_v + num_matched_j > len(cdr3_nucseq):
        ## some overlap!
        extra = num_matched_v + num_matched_j - len(cdr3_nucseq )
        fake_v_trim = extra//2 ## now deterministic
        fake_j_trim = extra - fake_v_trim
        num_matched_v -= fake_v_trim
        num_matched_j -= fake_j_trim

    assert num_matched_v + num_matched_j <= len(cdr3_nucseq)

    if num_matched_v + num_matched_j == len(cdr3_nucseq):
        nseq = ''
    else:
        nseq = cdr3_nucseq[num_matched_v:len(cdr3_nucseq)-num_matched_j]

    ncount = [1] * len(cdr3_nucseq)
    cdr3_nucseq_src = ['N'] * len(cdr3_nucseq)
    for i in range(num_matched_v):
        ncount[i] = 0
        cdr3_nucseq_src[i] = 'V'
    for i in range(num_matched_j):
        ncount[-1-i] = 0
        cdr3_nucseq_src[-1-i] = 'J'

    assert num_matched_v + num_matched_j + len(nseq) == len(cdr3_nucseq)

    v_trim = len(v_nucseq)-num_matched_v
    j_trim = len(j_nucseq)-num_matched_j

    assert len(cdr3_nucseq) == len(v_nucseq) + len(nseq) + len(j_nucseq) - ( v_trim + j_trim )

    #d_info = ''
    n_vj_insert = 0
    n_vd_insert = 0
    n_dj_insert = 0
    d0_trim = 0
    d1_trim = 0
    best_d_id = 0

    if ab == 'A':
        n_vj_insert = len(nseq)

    elif ab == 'B':
        ## look for one of the d-gene segments
        max_overlap = 0
        for d_id, d_nucseq in all_trbd_nucseq[organism].items():
            if force_d_id and d_id != force_d_id: continue
            for start in range(len(d_nucseq)):
                for stop in range(start,len(d_nucseq)):
                    overlap_seq = d_nucseq[start:stop+1]
                    if overlap_seq in nseq:
                        if len(overlap_seq)>max_overlap:
                            max_overlap = len(overlap_seq)
                            best_d_id = d_id
                            best_overlap_seq = overlap_seq
                            best_trim = (start,len(d_nucseq)-1-stop)

        if max_overlap: ## found a bit of d, although it might be bogus (eg 1 nt)
            pos = nseq.index( best_overlap_seq )
            for i in range(pos+num_matched_v,pos+num_matched_v+max_overlap):
                assert ncount[i] == 1
                ncount[i] = 0
                cdr3_nucseq_src[i] = 'D'
                nseq = nseq[:i-num_matched_v] + '+' + nseq[i+1-num_matched_v:]

            n_vd_insert = pos
            n_dj_insert = len(nseq) - pos - len(best_overlap_seq)
            d0_trim = best_trim[0]
            d1_trim = best_trim[1]

            expected_cdr3_nucseq_len = ( len(v_nucseq) + n_vd_insert +
                                         len(all_trbd_nucseq[organism][best_d_id]) + n_dj_insert +
                                         len(j_nucseq) -
                                         ( v_trim + d0_trim + d1_trim + j_trim ) )
            assert len(cdr3_nucseq) == expected_cdr3_nucseq_len


        else:
            best_d_id = 0
            n_vd_insert = 0
            n_dj_insert = 0
            d0_trim = 0
            d1_trim = 0




    if cdr3_protseq:
        assert 3*len(cdr3_protseq) == len(ncount)

        newseq = ''
        newseq_ncount = ''

        for i,a in enumerate(cdr3_protseq):
            nc = sum(ncount[3*i:3*i+3])
            newseq_ncount += repr(nc)
            if nc>1:
                newseq += a
            elif nc==1:
                newseq += a.lower()
            else:
                newseq += '-'

        ## output
        cdr3_protseq_masked = newseq[:]
        cdr3_protseq_new_nucleotide_countstring = newseq_ncount[:]
    else:## no protseq given (perhaps its an out of frame?)
        cdr3_protseq_masked = ''
        cdr3_protseq_new_nucleotide_countstring = ''

    new_nucseq = nseq[:]

    trims = ( v_trim, d0_trim, d1_trim, j_trim )
    inserts = ( best_d_id, n_vd_insert, n_dj_insert, n_vj_insert )

    ## new_nucseq spans the inserted nucleotide sequence and has '+' for D-nucleotides
    if return_cdr3_nucseq_src:
        return new_nucseq, cdr3_protseq_masked, cdr3_protseq_new_nucleotide_countstring, trims, inserts, cdr3_nucseq_src
    else:
        return new_nucseq, cdr3_protseq_masked, cdr3_protseq_new_nucleotide_countstring, trims, inserts




def add_masked_CDR3_sequences_to_tcr_dict( organism, vals ):
    ## this code is mostly taken from compute_probs.py
    va_gene = vals['va_gene']
    ja_gene = vals['ja_gene']
    vb_gene = vals['vb_gene']
    jb_gene = vals['jb_gene']
    cdr3a_protseq = vals['cdr3a']
    cdr3a_nucseq  = vals['cdr3a_nucseq']
    cdr3b_protseq = vals['cdr3b']
    cdr3b_nucseq  = vals['cdr3b_nucseq']

    a_junction_results = analyze_junction( organism, va_gene, ja_gene, cdr3a_protseq, cdr3a_nucseq )
    b_junction_results = analyze_junction( organism, vb_gene, jb_gene, cdr3b_protseq, cdr3b_nucseq )

    cdr3a_new_nucseq, cdr3a_protseq_masked, cdr3a_protseq_new_nucleotide_countstring,a_trims,a_inserts \
        = a_junction_results
    cdr3b_new_nucseq, cdr3b_protseq_masked, cdr3b_protseq_new_nucleotide_countstring,b_trims,b_inserts \
        = b_junction_results

    # from tcr_sampler.py:
    # trims = ( v_trim, d0_trim, d1_trim, j_trim )
    # inserts = ( best_d_id, n_vd_insert, n_dj_insert, n_vj_insert )

    assert a_trims[1] + a_trims[2] + a_inserts[0] + a_inserts[1] + a_inserts[2] + b_inserts[3] == 0
    assert a_inserts[3] == len( cdr3a_new_nucseq )

    ita = '+%d-%d'%(sum(a_inserts[1:]),sum(a_trims))
    itb = '+%d-%d'%(sum(b_inserts[1:]),sum(b_trims))

    vals[ 'cdr3a_protseq_masked'] = cdr3a_protseq_masked
    vals[ 'a_indels'] = ita
    vals[ 'cdr3a_new_nucseq' ] = cdr3a_new_nucseq
    vals[ 'cdr3b_protseq_masked'] = cdr3b_protseq_masked
    vals[ 'b_indels'] = itb
    vals[ 'cdr3b_new_nucseq' ] = cdr3b_new_nucseq


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

