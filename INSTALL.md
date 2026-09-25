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

# Install with performance optimization (FAISS)
pip install -e ".[performance]"

# Or install with batch integration support
pip install -e ".[batch]"

# Or install with scVI support (experimental)
pip install -e ".[scvi]"

# Or install everything
pip install -e ".[all]"
```

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

Already included in `environment.yml`. For manual installation:
```bash
# CPU version (recommended for most users)
mamba install -c conda-forge faiss-cpu

# GPU version (if you have CUDA-capable GPU)
mamba install -c conda-forge faiss-gpu
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
