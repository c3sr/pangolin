#pragma once

#include <vector>

#include "graph/allocator/cuda_zero_copy.hpp"

template <typename T>
using CUDAZeroCopyVector = std::vector<T, CUDAZeroCopyAllocator<T>>;