#pragma once

#include "pangolin/namespace.hpp"

PANGOLIN_NAMESPACE_BEGIN()

/*! \brief return (1, index) if search_val is in array between left and right, inclusive
return (0, -1) otherwise
*/
template<typename T>
__device__ static ulonglong2 binary_search(const T *const array, 
    size_t left,
    size_t right,
    const T search_val
) {
    while (left <= right) {
        size_t mid = (left + right) / 2;
        T val = array[mid];
        if (val < search_val) {
            left = mid + 1;
        } else if (val > search_val) {
            if (mid == 0) { // prevent rollover when mid = 0 and right becomes unsigned max
                break;
            } else {
                right = mid - 1;
            }
        } else { // val == search_val
            return make_ulonglong2(1, mid);
        }
    }
    return make_ulonglong2(0, (unsigned long long)(-1));
}


PANGOLIN_NAMESPACE_END()