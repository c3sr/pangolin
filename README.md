# Pangolin

[![Netlify Status](https://api.netlify.com/api/v1/badges/9996fcec-ff4e-4664-ae94-3734b469d5d9/deploy-status)](https://app.netlify.com/sites/pangolin-docs/deploys)

| Branch | Status |
|-|-|
| master  | [![Build Status](https://travis-ci.com/c3sr/pangolin.svg?token=oXpZxp44qzps6HC63xis&branch=master)](https://travis-ci.com/c3sr/pangolin)|

A header-only C++/CUDA library for GPU graph operations. [C++ API Documentation](https://pangolin-docs.netlify.app/).

## Getting Started

### Install CUDA

Most of pangolin only works with `nvcc`.

### Install NUMA (optional)

`sudo apt install libnuma-dev`

Include the pangolin headers in your code, and somewhere before using pangolin, call `pangolin::init()`

```c++
#include "pangolin.hpp"
#include "pangolin.cuh"

int main(int argc, char **argv) {
    pangolin::init();
    // your code here.
}
```

## Testing

```
mkdir -p build && cd build
make
make test
```

## Library Features

### GPU Graph Algorithms
* Triangle Counting
  * Anjur-Iyer: triangle counting algorithm created by Anjur and Iyer in ECE 508.
  * ~~Bisson-Fatica 2019: triangle counting algorithm from Bisson & Fatica~~
  * Edge-Binary: Edge-oriented count with binary search intersection
  * Edge-Dyn: Edge-oriented count with warp-granularity dynamic load balancing and cost-based binary/sequential search selection
  * ~~Edge-dynpar: Edge-oriented binary search count with dynamic parallelism~~
  * Edge-Dysel: Edge-oriented count with empirical-performance measurement and algorithm selection
  * Edge-Linear: Edge-oriented count with linear-search intersection
  * ~~Edge-Merge: Edge-oriented count with merge-path search~~
  * ~~Task-GPU: 2D decomposition count~~
  * Vertex-Bitvector: vertex-oriented bitvector search in shared memory
  * Vertex-Block-Binary: vertex-oriented binary intersection threadblock-per-row 
  * Vertex-Blocks-Binary: vertex-oriented binary intersection threadblocks-per-row 
    * ~~Vertex-Blocks-Cache-Block-Binary:~~
  * Vertex-CPU: CPU counters
  * Vertex-Dyn: Like Edge-Dyn but vertex-oriented
  * Vertex-Warp-Bitvector: vertex oriented count with warp-collaborative bit vectors
* K-truss
* broadcast: `include/pangolin/algorithm/broadcast.cuh`
  * shared memory and shuffle instructions
  * warp-collaborative and block-collaborative
  * non-primitive types
* elementwise: `include/pangolin/algorithm/elementwise.cuh`
  * Element-wise CSR matrix multiplication
* Grid-collaborative Fill: `include/pangolin/algorithm/fill.cuh`
* Load-Balance:
  * For a set of objects creating work-items, determine which object created each work item
* Merge-Path:
  * single-thread compute location where merge-path crosses diagonal
* Reduction: `include/pangolin/algorithm/search.cuh`
  * warp-collective sum/min/max with shuffle instructions
* Search: `include/pangolin/algorithm/search.cuh`
  * single-thread binary search `__device__` function: `serial_sorted_search_binary`
    * upper and lower bound
  * single-thread pointer-chasing linear search `__device__` function: `serial_sorted_search_linear`


Get started at [include/pangolin/algorithm].

### RcStream
A reference-counted `cudaStream_t`.
Automatically create, share, and destroy a single cudaStream_t, analogous to a `std::shared_ptr`.
Get started at [include/pangolin/cuda_cxx/rc_stream.hpp].

### Allocators
C++ stdlib allocators for CUDA device memory, CUDA host memory, and CUDA managed memory.
Get started at [include/pangolin/allocator].

### Dense containers
`Vector`s and `Buffer`s backed by C++ allocator
Get started at [include/pangolin/dense].

### Sparse containers
CSR and CSR+COO sparse matrices backed by `pangolin::Vector`
Get started at [include/pangolin/sparse].

### Buffers and Queues
* Double Buffer: `double_buffer.hpp` a fixed-size single-producer/single-consumer double buffer
* Bounded BUffer: `bounded_buffer.hpp` a fixed-size blocking multi-producer multi-consumer ring buffer

### System Topology Exploration
Built on top of `numa` and `nvidia-ml`, query the system topology to discover which GPUs, CPUs, and NUMA regions are associated.
Get started at [include/pangolin/topology].

### Controlling Logging

```c++
pangolin::logger::set_level(pangolin::logger::Level::ERR)
```

Allowed values are `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERR`, `CRITICAL`.

### API Documentation

API documentation is available at [pangolin-docs.netlify.com](https://pangolin-docs.netlify.com/).


## Building Pangolin from Source

### Prerequisites

| Dockerfile | cpu | CUDA | c++ | CMake | Builds |
|-|-|-|-|-|-|
| test_cuda92-ubuntu1804.Dockerfile         | amd64  | 9.2     | g++ 7.3.0   | 3.11.0 | &#9745; |
| test_cuda100-ubuntu1804.Dockerfile        | amd64  | 10.0    | g++ 7.3.0   | 3.11.0 | &#9745; |
| - | amd64 (Ubuntu 16.04) | 10.0.130 | g++ 5.4.0 | 3.14.3 | &#9745; |
|                                           | POWER9 | 9.2.148 | clang 5.0.0 | 3.12.0 | &#9745; |
|                                           | POWER9 | 9.2.148 | g++ ??? | 3.12.0 | ??? |
| test_cuda80-ubuntu1404.Dockerfile         | amd64  | 8.0.61  | g++ 4.8.4   | 3.11.0 | &#9745; |
| test_cuda80-ubuntu1404-clang38.Dockerfile | amd64  | 8.0.61  | clang 3.8.0 | 3.11.0 | (needs check) x: problem parsing Vector |
| test_cuda80-ubuntu1604.Dockerfile         | amd64  | 8.0.61  | g++ 5.4.0   | 3.11.0 | (needs check) x: problem parsing Vector |
| test_cuda92_ubuntu1604-clang5.Dockerfile  | amd64  | 9.2.148 | clang 5.0.0 | 3.11.0 | x: problem with simd intrinsics |
| - | amd64 | 9.2.148 | g++5.4.1 | 3.13.3 | x: problem with std::to_string in catch2 | 



1. Install CUDA

Instructions for installing CUDA on supported systems may be obtained from Nvidia's website.

2. Install CMake 3.13+

On x86 linux, CMake provides prebuilt binaries with a shell script.
On POWER, you will need to build CMake from source.
You can check your cmake version with `cmake --version`.
CMake will need to built with support for SSL.

3. (Optional) Install Doxygen/Graphviz for API documentation

Install doxygen and graphviz

    sudo apt install doxygen graphviz

If doxygen is installed, building pangolin will also create API documentation.

### Building on Ubuntu/Debian

Pangolin is a header-only library, but you can still build the tests

    mkdir -p build
    cd build
    cmake ..
    make
    make tests

## Using Pangolin in another CMake project

See [Pangolin_Example](https://github.com/c3sr/pangolin_example) for an example.

Pangolin may be used with CMake `add_subdirectory()`, or installed and used with CMake `find_package(pangolin CONFIG)`.
Pangolin exports the CMake `pangolin::pangolin` target.

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

Pangolin is a header-only library, so "installation" is a matter of copying pangolin's headers to a desired location.
Pangolin also includes a CMake config file for easy integration with other CMake projects.

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
target_link_libraries(... pangolin::pangolin)
```

## Running tests

Tests can be built and run with 

```
make
make test
```

Most tests require a GPU (those tests have the `gpu` label).
Some tests require MPI (those tests have the `mpi` label)

```
ctest -LE "gpu" # run tests that do not require a GPU
ctest -L "mpi" # run tests that require MPI
```

To run individual tests, you can do something like

```
make
test/test_csr
```

## Continuous Integration

We automatically build and test the following configurations.

| CI Platform | CUDA | NUMA | MPI |  Build | Test |
|-|-|-|-|-|
| Azure Pipelines | 10.1 | Yes | Yes | Yes | Yes |
| Azure Pipelines | 10.1 | No  | Yes | Yes | Yes |
| Azure Pipelines | 10.1 | Yes | No  | Yes | non-mpi |
| Azure Pipelines | 10.1 | No  | No  | Yes | non-mpi |

## Profiling

[CUDA metrics guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#metrics-reference)

### GPUs with CC <= 7.2

Non-interactive profiling is done by creating two different profiling files: a timeline file, and a metrics file.
These are created with two separate invocations of nvprof:

```
nvprof -o timeline.nvvp -f ./<exe> ...
nvprof -o metrics.nvvp -f --analysis-metrics ./<exe>
```
These files can be opened in nvvp.

File >> import >> nvprof >> single process

timeline data file: timeline.nvvp
Event/Metric data files: metrics.nvvp

### GPUs with CC > 7.2

On GPUs with CC > 7.2, some version of Nsight needs to be used.
Either open the NVIDIA Nsight Compute profiler and do it interactively, or generate a report and import it

    /usr/local/cuda/NsightCompute-1.0/nv-nsight-cu-cli -o profile -f ...

or

    /usr/local/cuda/NsightCompute-1.0/nv-nsight-cu&

`File` > `Open File` > `profile.nsight-cuprof-report`


#### Interactive Timeline
The Nvidia Nsight Eclipse Edition can generate a timeline

* Open Nsight Eclise Edition

On Ubuntu 18.04, there may be a conflict with the installed Java runtime (usually openjdk-11).

```
sudo apt install openjdk-8-jre
```

Then add the path to the java8 runtime to the top of `/usr/local/cuda/libnsight/nsight.ini` like so

```
-vm
/usr/lib/jvm/java-8-openjdk-amd64/jre/bin
```

* Run >> Profile Configurations
* Select "C/C++ Application" in the left panel
* Press the 'New' button near the top-left to create a new profiling configuration
* Give it a name near the top middle if you want
* Main tab >> Put the path to the CUDA binary in "C/C++ Application"
* Arguments tab >> put the command line arguments to the binary here

#### Non-interactive Timeline

Generate a timeline with nvprof

`nvprof -o timeline.nvvp -f ./mybin`

* Open Nsight Eclipse Edition
* File >> Import
  * CUDA `v` Nvprof
  * Put the timeline file in the timeline file box
  * Uncheck `Use fixed width segments for unified memory timeline`

#### Interactive Detailed Profile

* Open "NVIDIA Nsight Compute"
* "Create New Project" >> cancel
* "Connect" in top-left
  * Put the path to the application in "Application Excecutable"
  * Put the arguments in "Command Line Arguments"
  * Enable NVTX Support = Yes
* Launch
* Profile >> auto-profile

[profiler report source page](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#profiler-report-source-page)

Measuring Divergence:
"predicated-on thread instructions executed" should be 32x "instructions executed" for no divergence.

"predicated-on thread instructions executed": instructions executed: number of times an instruction was executed by a warp
Number of times the source (instruction) was executed by any active, predicated-on thread. For instructions that are executed unconditionally (i.e. without predicate), this is the number of active threads in the warp, multiplied with the respective Instructions Executed value. 

#### Non-interactive Detailed Profile

Generate a profile using something like `/usr/local/cuda/NsightCompute-1.0/nv-nsight-cu-cli -o ./my-binary`

* Open "NVIDIA Nsight Compute"
* File >> Open >> `<the profile file you generated>`

## Other

See [references](references) for some notes on references.

## Publications

```bibtex
@INPROCEEDINGS{8916285,
author={M. {Almasri} and O. {Anjum} and C. {Pearson} and Z. {Qureshi} and V. S. {Mailthody} and R. {Nagi} and J. {Xiong} and W. {Hwu}},
booktitle={2019 IEEE High Performance Extreme Computing Conference (HPEC)},
title={Update on k-truss Decomposition on GPU},
year={2019},
volume={},
number={},
pages={1-7},
}
```

```bibtex
@INPROCEEDINGS{8916547,
 author={C. {Pearson} and M. {Almasri} and O. {Anjum} and V. S. {Mailthody} and Z. {Qureshi} and R. {Nagi} and J. {Xiong} and W. {Hwu}},
 booktitle={2019 IEEE High Performance Extreme Computing Conference (HPEC)},
 title={Update on Triangle Counting on GPU},
 year={2019},
 volume={},
 number={},
 pages={1-7},
 } 
```

```bibtex
@INPROCEEDINGS{8547517,
author={V. S. {Mailthody} and K. {Date} and Z. {Qureshi} and C. {Pearson} and R. {Nagi} and J. {Xiong} and W. {Hwu}}, booktitle={2018 IEEE High Performance extreme Computing Conference (HPEC)},
title={Collaborative (CPU + GPU) Algorithms for Triangle Counting and Truss Decomposition},
year={2018},
volume={},
number={},
pages={1-7},

```