######################################################################################88
import argparse
from os.path import exists
import sys
import os

parser = argparse.ArgumentParser()

parser.add_argument('--config', help="configuration file *.yml", type=str)
parser.add_argument('--output_clones_file')
parser.add_argument('--input_clones_file') # option to skip the 10x parsing if we already have a clones file
parser.add_argument('--organism', choices=['mouse', 'human', 'mouse_gd', 'human_gd', 'human_ig','rhesus','rhesus_gd'], default = None)
#parser.add_argument('--n_components', type=int, default=50)
parser.add_argument('--filtered_contig_annotations_csvfile', help='Required unless --input_clones_file is present')
parser.add_argument('--consensus_annotations_csvfile', help='Not needed')
parser.add_argument('--no_tcrdists', action='store_true')
parser.add_argument('--no_kpca', action='store_true', help='Make the clones file '
                    'but dont compute the tcrdist distance matrix or run kernel PCA. '
                    'Useful for really large datasets; used in conjunction with '
                    'run_conga.py --no_kpca option')
parser.add_argument('--save_tcrdist_matrices', action='store_true')

# TCR representation selection flags (matching run_conga.py)
parser.add_argument('--use_kpca_tcrdist', action='store_true',
                    help='Force use of KernelPCA TCR representation (X_pca_tcr)'
                    ' instead of vectorized representation for supported organisms')
parser.add_argument('--kpca_reduction_limit', type=int, default=None,
                    help='Observation count at or above which the KernelPCA '
                    'reduction is not performed (default: 20000)')

parser.add_argument('--kpca_kernel')
parser.add_argument('--kpca_gaussian_kernel_sdev', default=100.0, type=float,
                    help='only used if kpca_kernel==\'gaussian\'')
parser.add_argument('--kpca_outfile')
parser.add_argument('--condense_clonotypes_by_tcrdist', action='store_true')
parser.add_argument('--tcrdist_threshold_for_condensing', type=float, default=50. )
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if len(sys.argv)==1:
    parser.print_help()
    sys.exit()

# update args specified in yml file
if args.config is not None:
    import yaml
    assert exists(args.config)
    yml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    for k, v in yml_args.items():
        if k in args.__dict__:
            args.__dict__[k] = v
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

if args.organism is None:
    print('Organism not specified. Add to --config file or specify with --organism')
    quit()

if args.input_clones_file is not None:
    assert exists(args.input_clones_file)
    assert exists(args.filtered_contig_annotations_csvfile is None) # doesn't make sense
else:
    assert exists(args.filtered_contig_annotations_csvfile)
    if args.consensus_annotations_csvfile is not None:
        assert exists(args.consensus_annotations_csvfile)

assert args.kpca_kernel in [None, 'gaussian'] #None means classic default

# put this after arg parsing because it's so dang slow
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) # so we can import conga
import conga
from conga import util
from conga.preprocess import (make_tcrdist_kernel_pcs_file_from_clones_file,
                              condense_clones_file_and_barcode_mapping_file_by_tcrdist,
                              resolve_tcr_representation)

from conga.tcrdist.make_10x_clones_file import make_10x_clones_file

# Resolve default values now that conga is imported
if args.kpca_reduction_limit is None:
    args.kpca_reduction_limit = util.KPCA_REDUCTION_LIMIT

# Flag conflict validation (Requirements 8.34, 8.19)
if args.use_kpca_tcrdist and args.no_kpca:
    print(f"ERROR: Conflicting flags --use_kpca_tcrdist and --no_kpca", file=sys.stderr)
    sys.exit(1)

if args.use_kpca_tcrdist and args.no_tcrdists:
    print(f"ERROR: Conflicting flags --use_kpca_tcrdist and --no_tcrdists", file=sys.stderr)
    sys.exit(1)


input_distfile = None

if args.input_clones_file is None:
    #### first make the clones file ####################################################
    stringent=True
    output_clones_file = args.filtered_contig_annotations_csvfile[:-4]+'_tcrdist_clones.tsv' \
                         if args.output_clones_file is None else args.output_clones_file
    #if args.condense_clonotypes_by_tcrdist:
    #    tmp_clones_filesuffix = '.uncondensed.tsv'
    #    output_clones_file += tmp_suffix # make a temporary version

    make_10x_clones_file(
        args.filtered_contig_annotations_csvfile,
        args.organism,
        output_clones_file,
        stringent=stringent,
        consensus_annotations_csvfile=args.consensus_annotations_csvfile,
        verbose=args.verbose
        )

    if args.condense_clonotypes_by_tcrdist:
        oldfile = output_clones_file
        output_clones_file = output_clones_file[:-4]+'_condensed.tsv'
        input_distfile = output_clones_file[:-4]+'_AB.dist'
        condense_clones_file_and_barcode_mapping_file_by_tcrdist(
            oldfile, output_clones_file, args.tcrdist_threshold_for_condensing, args.organism,
            output_distfile=input_distfile)


else:
    output_clones_file = args.input_clones_file

assert exists(output_clones_file)

# Read clones file to get clonotype count for path resolution
import pandas as pd
clones_df = pd.read_csv(output_clones_file, sep='\t', nrows=1)  # Just to get column info
clones_df = pd.read_csv(output_clones_file, sep='\t')
num_clones = len(clones_df)

# Handle error case: KernelPCA override above limit (Requirement 8.34)
if args.use_kpca_tcrdist and num_clones >= args.kpca_reduction_limit:
    print(f"ERROR: KernelPCA override requested (--use_kpca_tcrdist) but clonotype count "
          f"{num_clones} >= limit {args.kpca_reduction_limit}.", file=sys.stderr)
    print(f"Use --kpca_reduction_limit to raise the limit.", file=sys.stderr)
    sys.exit(1)

# Resolve which TCR representation path to use (Requirements 8.32-8.38)
tcr_representation = resolve_tcr_representation(
    organism=args.organism,
    num_obs=num_clones,
    request_kpca=args.use_kpca_tcrdist,
    request_exact_nbrs=args.no_kpca,
    kpca_reduction_limit=args.kpca_reduction_limit,
)

# Display path selection message
print(f"Setup analyzed {num_clones} clonotypes for organism '{args.organism}'")
print(f"TCR representation selected: {tcr_representation.reason}")

# Check if we should skip KernelPCA computation based on selected path
should_skip_kpca = (args.no_tcrdists or args.no_kpca or not tcr_representation.build_kpca)

if should_skip_kpca:
    if args.no_tcrdists or args.no_kpca:
        print(f'Skipping TCRdist calculations and kernel PCA per --no_tcrdists/--no_kpca flags')
    else:
        # Provide path-specific messaging (Requirements 8.32, 8.36)
        if tcr_representation.active == util.OBSM_KEY_VEC_TCR:
            print(f'Skipping TCRdist matrix and KernelPCA computation for supported organism {args.organism}.')
            print(f'The analysis pipeline will use vectorized TCR encoding (X_vec_tcr) by default.')
        elif tcr_representation.active == util.ACTIVE_REP_EXACT:
            print(f'Skipping TCRdist matrix and KernelPCA computation due to observation count >= limit.')
            print(f'The analysis pipeline will compute exact TCRdist neighbors on demand.')
else:
    print(f'Computing TCRdist matrix and KernelPCA for downstream analysis.')

#### Now compute the kernel PCs #######################################################

if should_skip_kpca:
    pass  # Already printed the appropriate message above
else:
    if args.save_tcrdist_matrices:
        output_distfile = output_clones_file[:-4]+'_AB.dist'
    else:
        output_distfile = None

    make_tcrdist_kernel_pcs_file_from_clones_file(
        output_clones_file,
        args.organism,
        kernel=args.kpca_kernel,
        outfile=args.kpca_outfile,
        gaussian_kernel_sdev=args.kpca_gaussian_kernel_sdev,
        input_distfile=input_distfile,
        output_distfile=output_distfile,
    )

print(f'Setup completed successfully!')
print(f'Use {output_clones_file} as the --clones_file argument to run_conga.py')

# Provide guidance based on the selected representation
if tcr_representation.active == util.OBSM_KEY_VEC_TCR:
    print(f'Analysis will use vectorized TCR encoding by default for {args.organism}.')
    if args.use_kpca_tcrdist:
        print(f'Note: --use_kpca_tcrdist was specified; analysis will compute KernelPCA instead.')
elif tcr_representation.active == util.OBSM_KEY_PCA_TCR:
    print(f'Analysis will use KernelPCA TCR representation.')
elif tcr_representation.active == util.ACTIVE_REP_EXACT:
    print(f'Analysis will compute exact TCRdist neighbors on demand.')
    print(f'Note: This requires compiled tcrdist_cpp binaries for practical runtimes.')

print('DONE')

