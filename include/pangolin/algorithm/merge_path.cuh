#pragma once

#include "bounds.hpp"
#include "pangolin/macro.h"

namespace pangolin {

/*! compute location of merge path at diagonal

    \return in A where the merge path crosses diagonal diag

    \tparam bounds upper or lower bound search
    \tparam AIt    random-access iterator type for array A
    \tparam BIt    random-access iterator type for array B

  The location in B may be computed as diag - result
 */
template <Bounds bounds, typename AIt, typename BIt>
PANGOLIN_HOST_DEVICE size_t merge_path(AIt A,            //!< [in] random access iterator array A
                                       const size_t aSz, //!< [in] size of A
                                       BIt B,            //!< [in] random access iterator array A
                                       const size_t bSz, //!< [in] size of A
                                       const size_t diag // the diagonal, numbered rightward  from the top-left corner

) {

  size_t begin = diag > bSz ? diag - bSz : 0; // max(0, diag-bSz)
  size_t end = diag < aSz ? diag : aSz;       // min(diag, aSz)

  while (begin < end) {
    const size_t mid = (begin + end) / 2;
    auto a = A[mid];
    auto b = B[diag - 1 - mid];
    bool pred = (Bounds::UPPER == bounds) ? a < b : !(b < a);
    if (pred)
      begin = mid + 1;
    else
      end = mid;
  }
  return begin;
}

} // namespace pangolin
