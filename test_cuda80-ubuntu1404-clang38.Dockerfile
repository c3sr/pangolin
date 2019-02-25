FROM nvidia/cuda:8.0-devel-ubuntu14.04 as builder

RUN apt-get update && apt-get install -y --no-install-suggests --no-install-recommends \
    curl \
    git \
&& rm -rf /var/lib/apt/lists/*

# install cmake
RUN mkdir -p /opt/cmake
RUN curl -SL https://cmake.org/files/v3.11/cmake-3.11.0-Linux-x86_64.tar.gz | tar -xz --strip-components=1 -C /opt/cmake
ENV PATH "/opt/cmake/bin:${PATH}"


# install clang
RUN mkdir -p /opt/clang
RUN curl -SL http://releases.llvm.org/3.8.0/clang+llvm-3.8.0-x86_64-linux-gnu-ubuntu-14.04.tar.xz | tar -xJ --strip-components=1 -C /opt/clang
ENV PATH "/opt/clang/bin:${PATH}"

# build pangolin
WORKDIR /build
COPY . /pangolin
RUN cmake /pangolin \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/pangolin/cmake/clang.toolchain \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CUDA_HOST_COMPILER=`which clang++`
RUN make -j`nproc` install || make VERBOSE=1 install
