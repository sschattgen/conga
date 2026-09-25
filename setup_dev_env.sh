#!/bin/bash
# CoNGA Development Environment Setup Script
# 
# This script sets up a complete development environment for CoNGA using mamba.
# It will create a conda environment, install dependencies, and compile C++ components.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="conga-dev"
PYTHON_VERSION="3.12"

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}CoNGA Development Environment Setup${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

# Check if mamba is installed
if ! command -v mamba &> /dev/null; then
    echo -e "${YELLOW}Warning: mamba not found${NC}"
    echo "Mamba is a faster alternative to conda."
    echo ""
    echo "Install options:"
    echo "  1. Install via conda: conda install -n base -c conda-forge mamba"
    echo "  2. Install Miniforge: https://github.com/conda-forge/miniforge"
    echo ""
    
    # Try to use conda instead
    if command -v conda &> /dev/null; then
        echo -e "${YELLOW}Falling back to conda...${NC}"
        CONDA_CMD="conda"
    else
        echo -e "${RED}Error: Neither mamba nor conda found!${NC}"
        echo "Please install conda or mamba first."
        exit 1
    fi
else
    CONDA_CMD="mamba"
    echo -e "${GREEN}✓ Found mamba${NC}"
fi

# Parse command line arguments
MINIMAL=false
FULL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            MINIMAL=true
            shift
            ;;
        --full)
            FULL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --minimal    Create minimal environment (core features only)"
            echo "  --full       Create full environment (all optional features)"
            echo "  --help       Show this help message"
            echo ""
            echo "Default: creates full environment"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine which environment file to use
if [ "$MINIMAL" = true ]; then
    ENV_FILE="environment-minimal.yml"
    echo -e "${YELLOW}Creating minimal environment...${NC}"
elif [ "$FULL" = true ] || [ "$MINIMAL" = false ]; then
    ENV_FILE="environment.yml"
    echo -e "${YELLOW}Creating full environment with all optional features...${NC}"
fi

# Check if environment already exists
if $CONDA_CMD env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        $CONDA_CMD env remove -n $ENV_NAME -y
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

# Create the environment
echo ""
echo -e "${GREEN}Step 1: Creating conda environment from ${ENV_FILE}...${NC}"
$CONDA_CMD env create -f $ENV_FILE

# Activate environment (for the commands below)
echo ""
echo -e "${GREEN}Step 2: Activating environment...${NC}"
eval "$($CONDA_CMD shell.bash hook)"
$CONDA_CMD activate $ENV_NAME

# Install conga in editable mode
echo ""
echo -e "${GREEN}Step 3: Installing CoNGA in editable mode...${NC}"
pip install -e .

# Compile C++ components
echo ""
echo -e "${GREEN}Step 4: Compiling C++ TCRdist components...${NC}"
if [ -d "tcrdist_cpp" ]; then
    cd tcrdist_cpp
    
    # Check if make is available
    if command -v make &> /dev/null; then
        echo "Using make to compile..."
        make
    else
        echo "Make not found, compiling manually..."
        mkdir -p bin
        g++ -O3 -std=c++11 -Wall -I ./include/ -o ./bin/find_neighbors ./src/find_neighbors.cc
        g++ -O3 -std=c++11 -Wall -I ./include/ -o ./bin/calc_distributions ./src/calc_distributions.cc
        g++ -O3 -std=c++11 -Wall -I ./include/ -o ./bin/find_paired_matches ./src/find_paired_matches.cc
    fi
    
    # Verify compilation
    if [ -f "bin/find_neighbors" ] && [ -f "bin/calc_distributions" ] && [ -f "bin/find_paired_matches" ]; then
        echo -e "${GREEN}✓ C++ components compiled successfully${NC}"
        chmod +x bin/*
    else
        echo -e "${RED}✗ C++ compilation may have failed${NC}"
        echo "Check the output above for errors."
    fi
    
    cd ..
else
    echo -e "${YELLOW}Warning: tcrdist_cpp directory not found${NC}"
fi

# Test installation
echo ""
echo -e "${GREEN}Step 5: Verifying installation...${NC}"

python -c "
import sys
try:
    import conga
    import scanpy
    import anndata
    import numpy
    import pandas
    print('✓ Core imports successful')
    
    # Check optional imports
    try:
        import faiss
        print('✓ FAISS available')
    except ImportError:
        print('  FAISS not available (optional)')
    
    try:
        import bbknn
        print('✓ BBKNN available')
    except ImportError:
        print('  BBKNN not available (optional)')
    
    try:
        import scvi
        print('✓ scVI-tools available')
    except ImportError:
        print('  scVI-tools not available (optional)')
    
    print('\nPython version:', sys.version.split()[0])
    print('NumPy version:', numpy.__version__)
    print('Pandas version:', pandas.__version__)
    print('Scanpy version:', scanpy.__version__)
    print('AnnData version:', anndata.__version__)
    
except Exception as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=====================================${NC}"
    echo -e "${GREEN}✓ Setup complete!${NC}"
    echo -e "${GREEN}=====================================${NC}"
    echo ""
    echo "To activate the environment, run:"
    echo -e "  ${YELLOW}${CONDA_CMD} activate ${ENV_NAME}${NC}"
    echo ""
    echo "To verify C++ compilation:"
    echo -e "  ${YELLOW}ls -la tcrdist_cpp/bin/${NC}"
    echo ""
    echo "To run tests:"
    echo -e "  ${YELLOW}pytest tests/${NC}"
    echo ""
    echo "See INSTALL.md for more information."
else
    echo ""
    echo -e "${RED}Setup encountered errors. Please check the output above.${NC}"
    exit 1
fi
