FROM python:3.12-slim

LABEL org.opencontainers.image.title="CoNGA" \
      org.opencontainers.image.description="Clonotype Neighbor Graph Analysis (CoNGA) for single-cell TCR/BCR-seq and RNA-seq data" \
      org.opencontainers.image.source="https://github.com/phbradley/conga"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies:
#  - build-essential/g++: compile the tcrdist_cpp executables (C++11)
#  - cmake/flex/bison: louvain (python-igraph's Louvain backend) builds its C
#    core from source via CMake on pip install; no manylinux wheel ships it
#  - imagemagick: svg -> png conversion for CoNGA plots (conga/convert_svg_to_png.py)
#  - git: only needed if installing extra packages from VCS; harmless to keep
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      flex \
      bison \
      imagemagick \
      git \
      ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/conga

# Copy the full repository (an editable install needs conga/ and scripts/
# present; .dockerignore keeps this from pulling in .git, test fixtures,
# docs, and other content not needed at runtime).
COPY . .

# Install CoNGA with the CPU performance extras (FAISS-CPU + fastcluster).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[performance]"

# Compile the optional C++ TCRdist executables (10-100x speedup for exact
# TCRdist calculations on large datasets). This is a soft dependency: conga
# falls back to the Python/sklearn implementation at runtime if the binaries
# are missing (see conga/util.py:tcrdist_cpp_available), so a build failure
# here should not break the image, but we fail loudly if it does happen since
# silently losing this is a real performance regression users may not notice.
RUN cd tcrdist_cpp && make && \
    ls bin/find_neighbors bin/calc_distributions bin/find_paired_matches

# Mount your data directory here when running the container, e.g.:
#   docker run -v /path/to/datasets:/data -it conga /bin/bash
VOLUME ["/data"]

CMD ["/bin/bash"]
