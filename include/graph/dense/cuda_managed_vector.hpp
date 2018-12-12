#pragma once

#include <vector>

#include "graph/allocator/cuda_managed.hpp"

template <typename T>
using CUDAManagedVector = std::vector<T, CUDAManagedAllocator<T>>;