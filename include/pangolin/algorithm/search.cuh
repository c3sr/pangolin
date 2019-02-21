#pragma once

#include "pangolin/namespace.hpp"

PANGOLIN_BEGIN_NAMESPACE()

/*! \brief return (1, index) if search_val is in array between left and right, inclusive
return (0, -1) otherwise
*/
template<typename T>
__device__ static ulonglong2 serial_sorted_search_binary(const T *const array, 
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


/*! \brief return the number of common elements between sorted litsts a and b
*/
template <typename T>
__device__ static size_t sorted_count_serial_linear(
    const T *const aBegin, //<! beginning of a
    const T *const aEnd, //<! end of a
    const T *const bBegin, //<! beginning of b
    const T *const bEnd //<! end of b
) {
    size_t count = 0;
    const T *ap = aBegin;
    const T *bp = bBegin;

    if (ap < aEnd && bp < bEnd) {

        bool loadA = false;
        bool loadB = false;
        T a = *ap;
        T b = *bp;
        
        while (ap < aEnd && bp < bEnd) {
            
            if (loadA) {
                a = *ap;
                loadA = false;
            }
            if (loadB) {
                b = *bp;
                loadB = false;
            }

          if (a == b) {
              ++count;
              ++ap;
              ++bp;
              loadA = true;
              loadB = true;
          }
          else if (a < b){
              ++ap;
              loadA = true;
          }
          else {
              ++bp;
              loadB = true;
          }
      }
    }
    return count;
}


PANGOLIN_END_NAMESPACE()