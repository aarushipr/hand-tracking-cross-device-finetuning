FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LLVM_VERSION=21

ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV VCPKG_DEFAULT_TRIPLET=x64-linux
ENV PATH="/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}"
ENV CC="clang-${LLVM_VERSION}"
ENV CXX="clang++-${LLVM_VERSION}"

SHELL ["/bin/bash", "-lc"]

RUN set -euxo pipefail \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      curl \
      wget \
      zip \
      unzip \
      tar \
      pkg-config \
      build-essential \
      ccache \
      python3-pip \
      gnupg \
      lsb-release \
      doxygen \
      graphviz \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null \
    && echo "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${LLVM_VERSION} main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      "clang-${LLVM_VERSION}" \
      "libclang-rt-${LLVM_VERSION}-dev" \
      "lld-${LLVM_VERSION}" \
      "llvm-${LLVM_VERSION}" \
      "llvm-${LLVM_VERSION}-tools" \
      "clang-format-${LLVM_VERSION}" \
      "clang-tidy-${LLVM_VERSION}" \
    && update-alternatives --install /usr/bin/cc cc "/usr/bin/clang-${LLVM_VERSION}" 100 \
    && update-alternatives --install /usr/bin/c++ c++ "/usr/bin/clang++-${LLVM_VERSION}" 100 \
    && update-alternatives --set cc "/usr/bin/clang-${LLVM_VERSION}" \
    && update-alternatives --set c++ "/usr/bin/clang++-${LLVM_VERSION}" \
    && python3 -m pip install --no-cache-dir cmake pre-commit \
    && command -v "clang-${LLVM_VERSION}" \
    && command -v cc \
    && command -v c++ \
    && command -v llvm-profdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
