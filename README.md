# Pangolin

## Controlling Logging

```c++
pangolin::logger::set_level(pangolin::Level::ERR)
```

Allowed values are `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERR`, `CRITICAL`.

## Building Pangolin from Source

### Prerequisites

| Dockerfile | cpu | CUDA | c++ | CMake | Builds |
|-|-|-|-|-|-|
| test_cuda92-ubuntu1804.Dockerfile         | amd64  | 9.2     | g++ 7.3.0   | 3.11.0 | &#9745; |
| test_cuda100-ubuntu1804.Dockerfile        | amd64  | 10.0    | g++ 7.3.0   | 3.11.0 | &#9745; |
|                                           | POWER9 | 9.2.148 | clang 5.0.0 | 3.12.0 | &#9745; |
| test_cuda80-ubuntu1404.Dockerfile         | amd64  | 8.0.61  | g++ 4.8.4   | 3.11.0 | &#9745; |
| test_cuda80-ubuntu1404-clang38.Dockerfile | amd64  | 8.0.61  | clang 3.8.0 | 3.11.0 | (needs check) x: problem parsing Vector |
| test_cuda80-ubuntu1604.Dockerfile         | amd64  | 8.0.61  | g++ 5.4.0   | 3.11.0 | (needs check) x: problem parsing Vector |
| test_cuda92_ubuntu1604-clang5.Dockerfile  | amd64  | 9.2.148 | clang 5.0.0 | 3.11.0 | x: problem with simd intrinsics |
| - | amd64 | 9.2.148 | g++5.4.1 | 3.13.3 | x: problem with std::to_string in catch2 | 


1. Install CUDA

Instructions for installing CUDA on supported systems may be obtained from Nvidia's website.

2. Install CMake 3.11+

On x86 linux, CMake provides prebuilt binaries with a shell script.
On POWER, you will need to build CMake from source.
You can check your cmake version with `cmake --version`.
CMake will need to built with support for SSL.

3. (Optional) Install Doxygen/Graphviz for API documentation

Install doxygen and graphviz

    sudo apt install doxygen graphviz

If doxygen is installed, building pangolin will also create API documentation.

### Building on Ubuntu/Debian

    mkdir -p build
    cd build
    cmake ..
    make
    make tests

This will produce two libraries: `pangolin32` and `pangolin64`.
Both have equivalent functionality, but use 32-bit and 64-bit values for graph vertex/edge IDs respectively.

## Using Pangolin in another CMake project

See [Pangolin_Example](https://github.com/c3sr/pangolin_example) for an example.

Pangolin may be used with CMake `add_subdirectory()`, or installed and used with CMake `find_package(pangolin CONFIG)`.
Pangolin exports two targets `pangolin::pangolin32` and `pangolin::pangolin64` for 32-bit or 64-bit integer types in graph IDs.

### As a Subdirectory

1. Add Pangolin as a git submodule

```
git submodule add https://github.com/c3sr/pangolin.git thirdparty/pangolin
cd thirdparty/pangolin
git checkout <hash, branch, etc>
```

2. Put `add_subdirectory(...)` in your CMakeLists

```cmake
# CMakeLists.txt
add_subdirectory(thirdparty/pangolin)
...
target_link_libraries(... pangolin::pangolin32)
```


### As an external Library

1. Clone and install pangolin to the desired location

```
git clone https://github.com/c3sr/pangolin.git
mkdir pangolin/build && cd pangolin/build
cmake .. -DCMAKE_INSTALL_PREFIX=<something>
make install
```

2. Use `-DCMAKE_PREFIX_PATH` CMake option and `find_package(pangolin CONFIG REQUIRED)` in your CMakeLists

```cmake
# CMakeLists.txt
find_package(pangolin CONFIG REQUIRED)
...
target_link_libraries(... pangolin::pangolin32)
```

## Getting Started

API documentation may be produced with `make docs` if Doxygen and Graphviz are installed.


## Profiling

[CUDA metrics guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#metrics-reference)

### GPUs with CC <= 7.2

nvprof -o timeline.nvvp -f ./gc ...
nvprof -o analysis.nvvp -f --analysis-metrics ./gc

### GPUs with CC > 7.2

On GPUs with CC > 7.2, some version of Nsight needs to be used.
Either open the NVIDIA Nsight Compute profiler and do it interactively, or generate a report and import it

    /usr/local/cuda/NsightCompute-1.0/nv-nsight-cu-cli -o profile -f ...

or

    /usr/local/cuda/NsightCompute-1.0/nv-nsight-cu&

`File` > `Open File` > `profile.nsight-cuprof-report`



## Other

See [references](references) for some notes on references.
