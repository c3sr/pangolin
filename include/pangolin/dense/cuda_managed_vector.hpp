#pragma once

#include <vector>

#include "pangolin/allocator/cuda_managed.hpp"

template <typename T>
using CUDAManagedVector = std::vector<T, CUDAManagedAllocator<T>>;