#pragma once

#include <vector>

#include "pangolin/allocator/cuda_zero_copy.hpp"

template <typename T>
using CUDAZeroCopyVector = std::vector<T, CUDAZeroCopyAllocator<T>>;