#pragma once

#include "bounds.hpp"
#include "pangolin/macro.h"

namespace pangolin {

/*! \brief find the lowest or highest place searchVal could be inserted into array and keep the ordering

    \tparam bounds UPPER or LOWER bound of searchVal in array
    \tparam It     random-access iterator into array
    \tparam T      type of element to search for
 */
template <Bounds bounds, typename It, typename T>
PANGOLIN_HOST_DEVICE size_t binary_search(It array,           //!< [in] array to search
                                          const size_t count, //!< [in] size of array
                                          const T searchVal   //!< [in] value to search for
) {
  size_t left = 0;
  size_t right = count;
  while (left < right) {
    const size_t mid = (left + right) / 2;
    T val = array[mid];
    bool pred = (Bounds::UPPER == bounds) ? !(searchVal < val) : val < searchVal;
    if (pred) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

} // namespace pangolin
