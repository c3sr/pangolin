# Graph

## Building graph from Source

### Prerequisites

1. Install CUDA

Instructions for installing CUDA on supported systems may be obtained from Nvidia's website.

2. Install CMake 3.12+

On x86 linux, CMake provides prebuilt binaries with a shell script.
On POWER, you will need to build CMake from source.
You can check your cmake version with `cmake --version`.

3. (Optional) Install Doxygen

    sudo apt install doxygen graphviz

If doxygen is installed, building graph will also create API documentation.

### Building on Ubuntu/Debian

    mkdir -p build
    cd build
    cmake ..
    make

This will produce two binaries: `tri32` and `tri64`.
Both have equivalent functionality, but use 32-bit and 64-bit values for graph vertex/edge IDs respectively.

## Running tri

    ./tri32 -h

## Citing

Our 2017 submission may be cited with the following bibtex entry

    @inproceedings{mailthody2018collaborative,
    title={Collaborative (CPU+ GPU) Algorithms for Triangle Counting and Truss Decomposition},
    author={Mailthody, Vikram S and Date, Ketan and Qureshi, Zaid and Pearson, Carl and Nagi, Rakesh and Xiong, Jinjun and Hwu, Wen-mei},
    booktitle={2018 IEEE High Performance extreme Computing Conference (HPEC)},
    pages={1--7},
    year={2018},
    organization={IEEE}
    }

## Other

See [references](references) for some notes on references.
