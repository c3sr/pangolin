FROM nvidia/cuda:8.0-devel-ubuntu16.04 as builder

RUN apt-get update && apt-get install -y --no-install-suggests --no-install-recommends \
    curl \
    git \
&& rm -rf /var/lib/apt/lists/*

# install cmake
RUN mkdir -p /opt/cmake
RUN curl -SL https://cmake.org/files/v3.11/cmake-3.11.0-Linux-x86_64.tar.gz | tar -xz --strip-components=1 -C /opt/cmake
ENV PATH "/opt/cmake/bin:${PATH}"

# build pangolin
WORKDIR /build
COPY . /pangolin
RUN cmake /pangolin -DCMAKE_BUILD_TYPE=Release
RUN make -j`nproc` install || make VERBOSE=1 install
