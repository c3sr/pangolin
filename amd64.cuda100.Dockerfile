FROM nvidia/cuda:10.0-devel-ubuntu18.04 as builder

RUN apt-get update && apt-get install -y --no-install-suggests --no-install-recommends \
    curl \
    git \
&& rm -rf /var/lib/apt/lists/*

# install cmake
RUN curl -SL https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.tar.gz | tar -xz --strip-components=1 -C /usr

# build pangolin
ENV PANGOLIN_INSTALL_DIR=/opt/pangolin
COPY . ~/.pangolin
WORKDIR ~/.pangolin
RUN mkdir build
WORKDIR build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PANGOLIN_INSTALL_DIR
RUN make -j`nproc` install

# keep the pangolin install
FROM nvidia/cuda:10.0-devel-ubuntu18.04
ENV PANGOLIN_INSTALL_DIR=/opt/pangolin
ENV PATH=$PANGOLIN_INSTALL_DIR/bin:$PATH
COPY --from=builder $PANGOLIN_INSTALL_DIR $PANGOLIN_INSTALL_DIR
RUN ldconfig -n $PANGOLIN_INSTALL_DIR/lib/libpangolin32.so
RUN ldconfig -n $PANGOLIN_INSTALL_DIR/lib/libpangolin64.so

# install python for the pangolin tools
RUN apt-get update && apt-get install -y --no-install-suggests --no-install-recommends \
    python \
&& rm -rf /var/lib/apt/lists/*
