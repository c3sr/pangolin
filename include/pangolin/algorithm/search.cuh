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

enum class SearchBounds { UPPER, LOWER };

/*! \brief return location of searchVal in array

    \tparam bounds UPPER or LOWER bound of searchVal in array
    \tparam It     random-access iterator into array
    \tparam T      type of element to search for
 */
template <SearchBounds bounds, typename It, typename T>
__device__ size_t binary_search(It array,           //!< [in] array to search
                                const size_t count, //!< [in] size of array
                                const T searchVal   //!< [in] value to search for
) {
  size_t left = 0;
  size_t right = count;
  while (left < right) {
    const size_t mid = (left + right) / 2;
    T val = array[mid];
    bool pred;
    switch (bounds) {
    case SearchBounds::UPPER:
      pred = val < searchVal;
      break;
    case SearchBounds::LOWER:
      pred = !(searchVal < val);
      break;
    }
    if (pred) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

} // namespace pangolin