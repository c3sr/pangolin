#pragma once

#include "pangolin/namespace.hpp"

namespace pangolin {

/*! \brief return a CUDA ulonglong2 (found, lower-bound) for search_val in array [left, right)
 */
template <typename T>
__device__ static ulonglong2 serial_sorted_search_binary(const T *const array, size_t left, size_t right,
                                                         const T search_val) {
  while (left < right) {
    size_t mid = (left + right) / 2;
    T val = array[mid];
    if (val < search_val) {
      left = mid + 1;
    } else if (val > search_val) {
      right = mid;
    } else { // val == search_val
      // return (found, location)
      return make_ulonglong2(1, mid);
    }
  }
  // return (not found, lower-bound of search_val into array)
  return make_ulonglong2(0, left);
}

} // namespace pangolin