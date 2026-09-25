# CoNGA Installation Guide

## Quick Start (Recommended)

### Prerequisites

Install [Mamba](https://mamba.readthedocs.io/) (faster conda alternative):
```bash
# If you have conda already:
conda install -n base -c conda-forge mamba

# Or install Miniforge with mamba built-in:
# https://github.com/conda-forge/miniforge
```

### Option 1: Full Development Environment (with all optional features)

```bash
# Create environment from environment.yml
mamba env create -f environment.yml

# Activate the environment
mamba activate conga-dev

# Install conga in editable mode
pip install -e .

# Compile C++ TCRdist components (highly recommended for performance)
cd tcrdist_cpp
make
cd ..

# Verify installation
python -c "import conga; print(conga.__version__)"
```

### Option 2: Minimal Environment (core features only)

```bash
# Create minimal environment
mamba env create -f environment-minimal.yml

# Activate the environment
mamba activate conga-dev

# Install conga in editable mode
pip install -e .

# Compile C++ components
cd tcrdist_cpp
make
cd ..
```

### Option 3: Install with specific optional features

```bash
# Create minimal environment
mamba env create -f environment-minimal.yml
mamba activate conga-dev

# Install with performance optimization (FAISS-CPU + fast clustering)
pip install -e ".[performance]"

# Or install with GPU performance (requires CUDA-capable GPU)
pip install -e ".[performance-gpu]"

# Or install with batch integration support
pip install -e ".[batch]"

# Or install with scVI support (experimental)
pip install -e ".[scvi]"

# Or install everything (includes CPU FAISS by default)
pip install -e ".[all]"

# Or install everything with GPU FAISS (requires CUDA-capable GPU)  
pip install -e ".[all-gpu]"
```

**Feature Groups:**
- `performance`: faiss-cpu + fastcluster (recommended for >10k cells)
- `performance-gpu`: faiss-gpu + fastcluster (for large datasets with CUDA GPU)
- `batch`: bbknn + batch correction tools
- `scvi`: scVI-tools for variational inference (experimental)
- `all`: All optional features with CPU FAISS
- `all-gpu`: All optional features with GPU FAISS (requires CUDA)

---

## Python Version Requirement

**CoNGA requires Python 3.12 or later.**

---

## Verifying Installation

### Test basic import
```bash
python -c "import conga; import scanpy; import anndata; print('CoNGA installation successful!')"
```

### Check C++ compilation
```bash
ls -la tcrdist_cpp/bin/
# Should see: find_neighbors, calc_distributions, find_paired_matches
```

### Run tests (if you have dev environment)
```bash
pytest tests/
```

---

## Optional Dependencies

### FAISS (for fast neighbor search on large datasets)

FAISS (Facebook AI Similarity Search) provides 5-100x performance improvements for neighbor search operations, especially on large datasets (>10k cells). CoNGA implements a tiered fallback system: faiss-gpu → faiss-cpu → sklearn.

**Automatic Installation (Recommended)**

FAISS-CPU is included in `environment.yml` by default:
```bash
mamba env create -f environment.yml  # Includes faiss-cpu>=1.7.4
```

**Manual Installation Options**

Choose based on your hardware and dataset size:

```bash
# CPU version (recommended for most users, works on all systems)
mamba install -c conda-forge faiss-cpu

# GPU version (requires CUDA-capable GPU, 10-50x faster on large datasets)
mamba install -c conda-forge faiss-gpu
```

**Selection Criteria**

| Use Case | Dataset Size | Hardware | Recommendation |
|----------|-------------|----------|----------------|
| Typical analysis | <50k cells | Any CPU | `faiss-cpu` (included by default) |
| Large dataset | >50k cells | Any CPU | `faiss-cpu` (significant speedup) |
| Very large dataset | >100k cells | CUDA GPU | `faiss-gpu` (maximum performance) |
| Cluster/HPC | Any size | Multiple GPUs | `faiss-gpu` with GPU scheduling |

**Performance Expectations**

| Backend | Relative Speed | Memory Usage | Hardware Requirements |
|---------|----------------|--------------|----------------------|
| sklearn (fallback) | 1x | High | Any system |
| faiss-cpu | 5-20x | Medium | Any system |
| faiss-gpu | 20-100x | Low | CUDA-capable GPU |

**Verification**

```bash
# Check FAISS installation and GPU availability
python -c "import conga.neighbors; conga.neighbors.get_backend_info()"
```

**Troubleshooting FAISS**

Common issues and solutions:

```bash
# Issue: ImportError for faiss-gpu
# Solution: Check CUDA compatibility
nvidia-smi  # Check CUDA version
# Install compatible faiss-gpu version

# Issue: FAISS not found during conda install  
# Solution: Check channel priority
mamba install -c conda-forge faiss-cpu --channel-priority strict

# Issue: GPU out of memory with faiss-gpu
# Solution: Control GPU memory usage with environment variables
export FAISS_OMP_NUM_THREADS=4  # Limit CPU threads
export CUDA_VISIBLE_DEVICES=0   # Use specific GPU
# Or falls back to CPU automatically

# Issue: Performance not improved with FAISS
# Solution: Ensure you're using datasets >10k cells
# FAISS overhead dominates on small datasets
```

**Environment Variables for GPU Control**

When using faiss-gpu, these variables can optimize performance:

```bash
# Limit CPU thread usage (recommended for shared systems)
export FAISS_OMP_NUM_THREADS=8

# Select specific GPU (multi-GPU systems)  
export CUDA_VISIBLE_DEVICES=0,1

# Control GPU memory allocation
export FAISS_GPU_MEM_FRACTION=0.9  # Use 90% of GPU memory
```

### BBKNN (for batch correction)

Already included in `environment.yml`. For manual installation:
```bash
mamba install -c bioconda bbknn
```

### scVI-tools (experimental integration learning)

Included via pip in `environment.yml`. For manual installation:
```bash
pip install scvi-tools
```

### SVG to PNG Conversion

CoNGA generates SVG plots and converts them to PNG. Install one of:

**ImageMagick (recommended, included in environment.yml):**
```bash
mamba install -c conda-forge imagemagick
# Or on macOS with Homebrew:
brew install imagemagick
```

**Inkscape:**
```bash
mamba install -c conda-forge inkscape
# Or on macOS with Homebrew:
brew install inkscape
```

**cairosvg:**
```bash
pip install cairosvg
```

---

## Compiling C++ TCRdist Components

The C++ implementation provides 10-100x speedup for TCR distance calculations.

### Requirements
- C++11 compatible compiler (g++, clang++)
- Make (optional, but convenient)

### Compilation

```bash
cd tcrdist_cpp

# Using make (recommended)
make

# Or manually compile each executable
g++ -O3 -std=c++11 -Wall -I ./include/ -o ./bin/find_neighbors ./src/find_neighbors.cc
g++ -O3 -std=c++11 -Wall -I ./include/ -o ./bin/calc_distributions ./src/calc_distributions.cc
g++ -O3 -std=c++11 -Wall -I ./include/ -o ./bin/find_paired_matches ./src/find_paired_matches.cc
```

### Verify compilation
```bash
ls -la bin/
# Should show three executables with execute permissions
./bin/find_neighbors --help  # Should run without error
```

### Troubleshooting C++ compilation

**Issue:** `g++: command not found`
```bash
# Install compiler via mamba (already in environment.yml)
mamba install -c conda-forge cxx-compiler

# Or on macOS, install Xcode Command Line Tools
xcode-select --install
```

**Issue:** Compilation errors
- Check compiler version: `g++ --version` (need 4.8+)
- Try using clang++: `make CXX=clang++`
- Check for proper include paths

---

## Platform-Specific Notes

### macOS
- Xcode Command Line Tools or conda cxx-compiler required
- ImageMagick path may need adjustment in `conga/convert_svg_to_png.py`
- Apple Silicon (M1/M2): Most dependencies work natively, FAISS has native support

### Linux
- Should work out of box with environment.yml
- Ensure C++ compiler installed: `sudo apt install build-essential` (Ubuntu/Debian)

### Windows
- Use WSL2 (Windows Subsystem for Linux) recommended
- Or use MinGW-w64 for C++ compilation
- Native Windows support experimental

---

## Updating the Environment

### Update all packages
```bash
mamba activate conga-dev
mamba update --all
```

### Update conga after pulling new code
```bash
git pull
pip install -e . --upgrade
```

### Recompile C++ components after updates
```bash
cd tcrdist_cpp
make clean
make
```

---

## Environment Management

### Export your environment
```bash
mamba env export > environment-snapshot.yml
```

### Remove environment
```bash
mamba deactivate
mamba env remove -n conga-dev
```

### List environments
```bash
mamba env list
```

---

## Development Installation

For contributing to CoNGA development:

```bash
# Clone the repository
git clone https://github.com/phbradley/conga.git
cd conga

# Create development environment with all tools
mamba env create -f environment.yml
mamba activate conga-dev

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Compile C++
cd tcrdist_cpp && make && cd ..

# Run tests
pytest tests/

# Run linting
ruff check conga/
black --check conga/
```

---

## Troubleshooting

### Import errors after installation
```bash
# Make sure you're in the right environment
mamba activate conga-dev

# Verify conga is importable
python -c "import conga; print(conga.__file__)"

# Reinstall in editable mode
pip install -e . --force-reinstall --no-deps
```

### FAISS not found
```bash
# Install FAISS
mamba install -c conda-forge faiss-cpu

# Or install without FAISS (falls back to sklearn)
# CoNGA will work but slower on large datasets
```

### C++ executables not found
```bash
# Compile them
cd tcrdist_cpp && make

# Verify they exist and are executable
ls -la tcrdist_cpp/bin/
chmod +x tcrdist_cpp/bin/*
```

### Memory issues with large datasets
- Use FAISS version (much more memory efficient)
- Subset data before analysis
- Increase available RAM
- Use a compute cluster or cloud instance

---

## Getting Help

- **GitHub Issues:** https://github.com/phbradley/conga/issues
- **Documentation:** See README.md and example notebooks
- **Paper:** https://www.nature.com/articles/s41587-021-00989-2

---

## Minimal Requirements (without conda/mamba)

If you prefer pip-only installation:

```bash
# Python 3.12+ required
python -m venv conga-env
source conga-env/bin/activate  # On Windows: conga-env\Scripts\activate

# Install from pyproject.toml
pip install -e .

# Or with all features
pip install -e ".[all]"

# You'll still need:
# - C++ compiler for tcrdist_cpp
# - ImageMagick/Inkscape for SVG conversion
```

Note: conda/mamba installation is strongly recommended as it handles all system dependencies (C++ compiler, ImageMagick, etc.) automatically.
