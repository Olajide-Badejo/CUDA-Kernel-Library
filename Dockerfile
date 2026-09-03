# The pinned build container: a machine with the exact assembler this project's
# instruction level evidence was recorded with.
#
# Why it exists. experiments/sass/golden holds the SASS of the top kernel, and
# scripts/sass_diff.py refuses to compare a capture made by a different toolkit,
# because two nvcc releases disagree about instruction selection and the diff
# would be noise rather than a regression. The golden was assembled by nvcc
# 13.3.73, which is what nvidia/cuda:13.3.1-devel-ubuntu24.04 carries, so that
# image and this file are what the sass-diff job and anyone regenerating the
# golden should use.
#
# It is pinned by digest, not by tag. A tag is a moving name: nvidia rebuilds
# 13.3.1-devel-ubuntu24.04 whenever a base package changes, and a rebuild that
# moved ptxas would turn the SASS gate red for a reason that has nothing to do
# with this repository.
#
# Build and use:
#   docker build -t ckl-build .
#   docker run --rm -v "$PWD":/src -w /src ckl-build \
#       cmake -S . -B build-container -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120-real
#   docker run --rm -v "$PWD":/src -w /src ckl-build cmake --build build-container
#
# Running the tests needs a GPU inside the container, which needs the NVIDIA
# container toolkit and --gpus all. Compiling, capturing SASS and diffing it
# against the golden need no GPU at all.

FROM nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87

# The container's host compiler is GCC 13.3, which nvcc 13.3 accepts. The
# machine this project is developed on defaults to GCC 15.2, whose libstdc++
# headers use `if consteval` in a way nvcc 13.3's frontend rejects, so builds
# there pass -DCMAKE_CUDA_HOST_COMPILER=g++-14. Nothing has to be passed here.
#
# gfortran and liblapack-dev are only needed for -DCKL_BUILD_FORTRAN=ON, and
# they are in the image so that cell of the matrix does not need a second one.
# The apt lists are removed in the same layer they are created in.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ccache \
        cmake \
        g++ \
        gfortran \
        git \
        liblapack-dev \
        make \
        ninja-build \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# CUTLASS and GoogleTest arrive through FetchContent at configure time, so a
# container build needs the network unless FETCHCONTENT_BASE_DIR points at a
# populated directory. Mount one to work offline.
ENV CMAKE_GENERATOR=Ninja

WORKDIR /src
