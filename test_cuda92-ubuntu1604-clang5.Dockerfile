FROM nvidia/cuda:9.2-devel-ubuntu16.04 as builder

RUN apt-get update && apt-get install -y --no-install-suggests --no-install-recommends \
    curl \
    git \
    python \
&& rm -rf /var/lib/apt/lists/*

# install cmake
RUN mkdir -p /opt/cmake
RUN curl -SL https://cmake.org/files/v3.11/cmake-3.11.0-Linux-x86_64.tar.gz | tar -xz --strip-components=1 -C /opt/cmake
ENV PATH "/opt/cmake/bin:${PATH}"

# install clang 5.0.0
RUN mkdir -p /opt/clang
# RUN curl -SL http://releases.llvm.org/5.0.0/clang+llvm-5.0.0-linux-x86_64-ubuntu16.04.tar.xz \
#     | tar -xJ --strip-components=1 -C /opt/clang
# RUN ls /opt/clang
# ENV PATH "/opt/clang/bin:${PATH}"

# download llvm
RUN mkdir -p /tmp/clang
RUN curl -SL http://releases.llvm.org/5.0.0/llvm-5.0.0.src.tar.xz \
    | tar -xJ -C /tmp/clang
# download clang
RUN mkdir -p /tmp/clang/llvm-5.0.0.src/tools/clang
RUN curl -SL http://releases.llvm.org/5.0.0/cfe-5.0.0.src.tar.xz \
    | tar -xJ --strip-components=1 -C /tmp/clang/llvm-5.0.0.src/tools/clang
# download openmp
RUN mkdir -p /tmp/clang/llvm-5.0.0.src/projects/openmp
RUN curl -SL http://releases.llvm.org/5.0.0/openmp-5.0.0.src.tar.xz \
    | tar -xJ --strip-components=1 -C /tmp/clang/llvm-5.0.0.src/projects/openmp
WORKDIR /tmp/clang
RUN cmake llvm-5.0.0.src \
    -DCMAKE_INSTALL_PREFIX=/opt/clang \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=Native
RUN nice -n20 make -j`nproc` install
ENV PATH "/opt/clang/bin:${PATH}"

# build pangolin
WORKDIR /build
COPY . /pangolin
RUN cmake /pangolin -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/pangolin/cmake/clang.toolchain \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CUDA_HOST_COMPILER=`which clang++`
RUN make -j`nproc` install || make VERBOSE=1 install
