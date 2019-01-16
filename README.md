# Graph

## Building Pangolin from Source

### Prerequisites

1. Install CUDA

Instructions for installing CUDA on supported systems may be obtained from Nvidia's website.

2. Install CMake 3.12+

On x86 linux, CMake provides prebuilt binaries with a shell script.
On POWER, you will need to build CMake from source.
You can check your cmake version with `cmake --version`.
CMake will need to built with support for SSL.

3. (Optional) Install Doxygen

    sudo apt install doxygen graphviz

If doxygen is installed, building graph will also create API documentation.

### Building on Ubuntu/Debian

    mkdir -p build
    cd build
    cmake ..
    make

This will produce two libraries: `pangolin32` and `pangolin64`.
Both have equivalent functionality, but use 32-bit and 64-bit values for graph vertex/edge IDs respectively.

## Using Pangolin with CMake

See [Pangolin_Example](https://github.com/c3sr/pangolin_example) for an example.

Pangolin may be used with CMake `add_subdirectory()`, or installed and used with CMake `find_package(pangolin CONFIG)`.
Pangolin exports two targets `pangolin::pangolin32` and `pangolin::pangolin64` for 32-bit or 64-bit integer types in graph IDs.

## Getting Started

API documentation may be produced with `make docs` if Doxygen and Graphviz are installed.

## Other

See [references](references) for some notes on references.
