#pragma once

#include "pangolin/utilities.hpp"

namespace pangolin {

/*! cudaMalloc space for n Ts

\tparam T the element type

 */
template <typename T>
T *cuda_malloc(const size_t n //<! [in] the number of elements to allocate
) {
  T *ret = nullptr;
  CUDA_RUNTIME(cudaMalloc(&ret, n * sizeof(T)));
  return ret;
}

} // namespace pangolin