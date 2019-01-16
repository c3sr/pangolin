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

## Other

See [references](references) for some notes on references.
