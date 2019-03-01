#pragma once

#include <cub/cub.cuh>

#include "pangolin/atomic_add.cuh"

namespace pangolin {

/*! \brief return the number of common elements between sorted litsts a and b
 */
template <typename T>
__device__ static size_t serial_sorted_count_linear(const T *const aBegin, //!< beginning of a
                                                    const T *const aEnd,   //!< end of a
                                                    const T *const bBegin, //!< beginning of b
                                                    const T *const bEnd    //!< end of b
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
      } else if (a < b) {
        ++ap;
        loadA = true;
      } else {
        ++bp;
        loadB = true;
      }
    }
  }
  return count;
}

/*! \brief return 1 if search_val is in array between [left, right). return 0 otherwise
 */
template <typename T>
__device__ static uint8_t serial_sorted_count_binary(const T *const array, size_t left, size_t right,
                                                     const T search_val) {
  while (left < right) {
    size_t mid = (left + right) / 2;
    T val = array[mid];
    if (val < search_val) {
      left = mid + 1;
    } else if (val > search_val) {
      right = mid;
    } else { // val == search_val
      return 1;
    }
  }
  return 0;
}

/*! \brief grid cooperative count of elements in A that appear in B

    @param[inout] count pointer to the count. caller should initialize to 0.
    \param        A     the array of needles
    \param        aSz   the number of elements in A
    \param        B     the haystack
    \param        bSz   the number of elements in B
*/
template <size_t BLOCK_DIM_X, typename T>
__device__ void grid_sorted_count_binary(uint64_t *count, const T *const A, const size_t aSz, const T *const B,
                                         const size_t bSz) {

  // Specialize BlockReduce for a 1D block of 128 threads on type int
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduce::TempStorage tempStorage;

  int gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;

  uint64_t threadCount = 0;

  for (size_t i = gx; i < aSz; i += gridDim.x * BLOCK_DIM_X) {
    // printf("looking for %d from 0 to %lu\n", A[i], bSz);
    threadCount += serial_sorted_count_binary(B, 0, bSz, A[i]);
  }

  // aggregate all counts found by this block
  uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

  // Add to total count
  if (0 == threadIdx.x) {
    // printf("found %lu\n", aggregate);
    atomicAdd(count, aggregate);
  }
}

/*! \brief warp cooperative count of elements in A that appear in B

    @param[inout] count           pointer to the count. caller should initialize to 0.
    @param[in]    A               array A
    @param[in]    aSz             the number of elements in A
    @param[in]    B               array B
    @param[in]    bSz             the number of elements in B
    \tparam       WARPS_PER_BLOCK the number of warps in the calling threadblock

    The calling threadblock should be made up of a number of complete warps.
    The longer array is searched in parallel for elements from the shorter array using a binary search.
*/
template <size_t WARPS_PER_BLOCK, typename T>
__device__ void warp_sorted_count_binary(uint64_t *count, const T *const A, const size_t aSz, const T *const B,
                                         const size_t bSz) {

  typedef cub::WarpReduce<uint64_t> WarpReduce;
  __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];

  const int warpIdx = threadIdx.x / 32; // which warp in thread block
  const int laneIdx = threadIdx.x % 32; // which thread in warp

  uint64_t threadCount = 0;
  if (aSz < bSz) {
    for (const T *const u = A + laneIdx; u < A + aSz; u += 32) {
      threadCount += pangolin::serial_sorted_count_binary(B, 0, bSz, *u);
    }
  } else {
    for (const T *const u = B + laneIdx; u < B + bSz; u += 32) {
      threadCount += pangolin::serial_sorted_count_binary(A, 0, aSz, *u);
    }
  }

  uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);

  if (laneIdx == 0) {
    *count = aggregate;
  }
}

} // namespace pangolin