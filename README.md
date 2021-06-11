# Clonotype Neighbor Graph Analysis (CoNGA) developmental version

Please see https://github.com/phbradley/conga for the latest version of CoNGA
and instructions.

This version incorporates the OLGA package find the generational probabilities
that are used by CoNGA as a TCR feature. It works but due to these calculations
is very slow on large datasets. If you stumble across this and you're interested 
in using Pgen as a feature, let us know! 

OLGA needs to be installed within your CoNGA conda environment
```
conda activate conga_env
pip install olga
```

