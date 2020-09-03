import argparse
from os.path import exists
import sys
import os

parser = argparse.ArgumentParser()

parser.add_argument('--output_clones_file')
parser.add_argument('--organism', choices=['mouse', 'human', 'mouse_gd', 'human_gd', 'human_ig'], required=True)
#parser.add_argument('--n_components', type=int, default=50)
parser.add_argument('--filtered_contig_annotations_csvfile', required=True)
parser.add_argument('--consensus_annotations_csvfile')

args = parser.parse_args()
assert exists(args.filtered_contig_annotations_csvfile)
if args.consensus_annotations_csvfile is not None:
    assert exists(args.consensus_annotations_csvfile)

# put this after arg parsing because it's so dang slow
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ) # so we can import conga
import conga
from conga.preprocess import make_tcrdist_kernel_pcs_file_from_clones_file
from conga.tcrdist.make_10x_clones_file import make_10x_clones_file



#### first make the clones file ###########################################################################
stringent=True
output_clones_file = args.filtered_contig_annotations_csvfile[:-4]+'_tcrdist_clones.tsv' \
                     if args.output_clones_file is None else args.output_clones_file

make_10x_clones_file(args.filtered_contig_annotations_csvfile, args.organism, output_clones_file, stringent=stringent,
                     consensus_annotations_csvfile=args.consensus_annotations_csvfile)

assert exists(output_clones_file)

#### Now compute the kernal PCs #############################################################3
make_tcrdist_kernel_pcs_file_from_clones_file( output_clones_file, args.organism )

print(f'If this all worked you should be able to pass {output_clones_file} as the --clones_file argument to run_conga.py')
print('DONE')

