set -x -e

source ci/env.sh

cd ${TRAVIS_BUILD_DIR}

which g++
which nvcc
which cmake

g++ --version
nvcc --version
cmake --version

mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make VERBOSE=1 