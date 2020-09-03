#This is nearly working. Need to figure out how to get it back in.
# cd /Users/sschattg/conga-dev
from os.path import exists
from os import system
import os.path
import olga.load_model as load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen
from . import util


#Define the files for loading in generative model/data
beta_params_file_name = util.path_to_olga + 'default_models/human_T_beta/model_params.txt'
beta_marginals_file_name = util.path_to_olga + 'default_models/human_T_beta/model_marginals.txt'
beta_V_anchor_pos_file = util.path_to_olga + 'default_models/human_T_beta/V_gene_CDR3_anchors.csv'
beta_J_anchor_pos_file = util.path_to_olga + 'default_models/human_T_beta/J_gene_CDR3_anchors.csv'

alpha_params_file_name = util.path_to_olga + 'default_models/human_T_beta/model_params.txt'
alpha_marginals_file_name = util.path_to_olga + 'default_models/human_T_beta/model_marginals.txt'
alpha_V_anchor_pos_file = util.path_to_olga + 'default_models/human_T_beta/V_gene_CDR3_anchors.csv'
alpha_J_anchor_pos_file = util.path_to_olga + 'default_models/human_T_beta/J_gene_CDR3_anchors.csv'

mus_beta_params_file_name = util.path_to_olga + 'default_models/mouse_T_beta/model_params.txt'
mus_beta_marginals_file_name = util.path_to_olga + 'default_models/mouse_T_beta/model_marginals.txt'
mus_beta_V_anchor_pos_file = util.path_to_olga + 'default_models/mouse_T_beta/V_gene_CDR3_anchors.csv'
mus_beta_J_anchor_pos_file = util.path_to_olga + 'default_models/mouse_T_beta/J_gene_CDR3_anchors.csv'

#Load models
beta_genomic_data = load_model.GenomicDataVDJ()
beta_genomic_data.load_igor_genomic_data(beta_params_file_name, beta_V_anchor_pos_file, beta_J_anchor_pos_file)
beta_generative_model = load_model.GenerativeModelVDJ()
beta_generative_model.load_and_process_igor_model(beta_marginals_file_name)

alpha_genomic_data = load_model.GenomicDataVDJ()
alpha_genomic_data.load_igor_genomic_data(alpha_params_file_name, alpha_V_anchor_pos_file, alpha_J_anchor_pos_file)
alpha_generative_model = load_model.GenerativeModelVDJ()
alpha_generative_model.load_and_process_igor_model(alpha_marginals_file_name)

mus_beta_genomic_data = load_model.GenomicDataVDJ()
mus_beta_genomic_data.load_igor_genomic_data(mus_beta_params_file_name, mus_beta_V_anchor_pos_file, mus_beta_J_anchor_pos_file)
mus_beta_generative_model = load_model.GenerativeModelVDJ()
mus_beta_generative_model.load_and_process_igor_model(mus_beta_marginals_file_name)

#Process model/data for pgen computation by instantiating GenerationProbabilityVDJ
beta_pgen_model = pgen.GenerationProbabilityVDJ(beta_generative_model, beta_genomic_data)
alpha_pgen_model = pgen.GenerationProbabilityVDJ(alpha_generative_model, alpha_genomic_data)
mus_beta_pgen_model = pgen.GenerationProbabilityVDJ(mus_beta_generative_model, mus_beta_genomic_data)


def beta_pgen_model, alpha_pgen_model, mus_beta_pgen_model



#human
#tcrs_test = [ ( ('TRAV8-6*01' , 'TRAJ52*01' , 'CAHTNAGGTSYGKLTF',  'tgtgctcatactaatgctggtggtactagctatggaaagctgacattt' ), 
#	('TRBV6-6*01', 'TRBJ1-6*01', 'CASSPYRVASPLHF','tgtgccagcagtccctacagggtagcttcacccctccacttt') ) ]

