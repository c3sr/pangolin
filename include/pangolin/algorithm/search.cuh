#pragma once

#include "bounds.hpp"
#include "pangolin/macro.h"
#include "pangolin/namespace.hpp"

namespace pangolin {

/*! \brief return a CUDA ulonglong2 (found, lower-bound) for search_val in array [left, right)
 */
template <typename T>
PANGOLIN_HOST_DEVICE static ulonglong2 serial_sorted_search_binary(const T *const array, size_t left, size_t right,
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

/*! \brief return a CUDA ulonglong2 (found, upper-bound) for search_val between [begin, end)

    upper-bound is a uintptr_t that can be converted to the T*.
    the search_val array must be sorted in increasing order between [begin, end)
    Search method is a linear scan through array
 */
template <typename T>
__device__ static ulonglong2
serial_sorted_search_linear(const T *const begin, //!< [in] beginning of array to search through
                            const T *const end,   //!< [in] end of array to search through
                            const T searchVal     //!< [in] value to search for
) {
  T *p = nullptr;
  for (p = begin; p < end; ++p) {
    if (*p == searchVal) {
      return make_ulonglong2(1, uintptr_t(p));
    }
    if (*p > searchVal) {
      return make_ulonglong2(0, uintptr_t(p));
    }
  }
  return make_ulonglong2(0, uintptr_t(p));
}

} // namespace pangolin
