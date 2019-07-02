#pragma once

#include <cub/cub.cuh>

#include "binary_search.cuh"
#include "pangolin/atomic_add.cuh"
#include "search.cuh"

namespace pangolin {

/*! \brief return the number of common elements between sorted lists a and b
 */
template <typename T>
__device__ static uint64_t serial_sorted_count_linear(const T *const aBegin, //!< beginning of a
                                                      const T *const aEnd,   //!< end of a
                                                      const T *const bBegin, //!< beginning of b
                                                      const T *const bEnd    //!< end of b
) {
  uint64_t count = 0;
  const T *ap = aBegin;
  const T *bp = bBegin;

  bool loadA = true;
  bool loadB = true;

  while (ap < aEnd && bp < bEnd) {

    T a, b;

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
  return count;
}

/*! \brief return the number of common elements between sorted lists A and B
 */
template <typename T>
__device__ static size_t serial_sorted_count_linear(const T *const A, //!< beginning of a
                                                    const size_t aSz,
                                                    const T *const B, //!< beginning of b
                                                    const size_t bSz) {
  return serial_sorted_count_linear(A, &A[aSz], B, &B[bSz]);
}

/*! \brief return 1 if search_val is in array between [left, right). return 0 otherwise

  array must be sorted in increasing order between [left, right)
  search method is binary search
 */
template <typename T>
__device__ static uint8_t serial_sorted_count_binary(const T *const array, //<! [in] array to search through
                                                     size_t left,          //<! [in] lower bound of search
                                                     size_t right,         //<! [in] upper bound of search
                                                     const T search_val    //<! [in] value to search for
) {
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

/*! \brief return 1 if search_val is between [begin, end). return 0 otherwise

    array must be sorted in increasing order between [begin, end)
    Search method is a linear scan through array
 */
template <typename T>
__device__ static uint8_t
serial_sorted_count_linear(const T *const begin, //<! [in] beginning of array to search through
                           const T *const end,   //<! [in] end of array to search through
                           const T searchVal     //<! [in] value to search for
) {
  for (T *p = begin; p < end; ++p) {
    T checkVal = *p;
    if (checkVal == searchVal) {
      return 1;
    }
    // early exit if we've searched past searchVal
    if (checkVal > searchVal) {
      return 0;
    }
  }
  return 0;
}

/*! \brief return 1 if search_val is in array between [left, right). return 0 otherwise

    array must be sorted in increasing order between [left, right)
    Search method is a linear scan through array
 */
template <typename T>
__device__ static uint8_t serial_sorted_count_linear(const T *const array, //<! [in] array to search through
                                                     size_t left,          //<! [in] lower bound of search
                                                     size_t right,         //<! [in] upper bound of search
                                                     const T search_val    //<! [in] value to search for
) {
  for (size_t searchIdx = left; searchIdx < right; ++searchIdx) {
    if (array[searchIdx] == search_val) {
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

  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
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

    \tparam       C               coarsening factor: elements of A per thread
    \tparam       WARPS_PER_BLOCK the number of warps in the calling threadblock
    \tparam       reduce          Reduce the result into lane 0
    \return                       The count found by the warp (in lane 0 only)

    The calling threadblock should be made up of a number of complete warps.
    Each thread searches for C elements of A in B.
    First, a binary search is used to find the lower bound of the chunk in B.
    Then, a sequential count of matching elements is used.
    Lane 0 in the warp returns the complete value
*/
template <size_t C, size_t WARPS_PER_BLOCK, typename T, bool reduce = true>
__device__ __forceinline__ uint64_t warp_sorted_count_binary(const T *const A, //!< [in] array A
                                                             const size_t aSz, //!< [in] the number of elements in A
                                                             const T *const B, //!< [in] array B
                                                             const size_t bSz  //!< [in] the number of elements in B
) {

  static_assert(C != 0, "expect at least 1 element per thread");

  const int warpIdx = threadIdx.x / 32; // which warp in thread block
  const int laneIdx = threadIdx.x % 32; // which thread in warp

  uint64_t threadCount = 0;

  // cover entirety of A with warp
  for (size_t i = laneIdx * C; i < aSz; i += 32 * C) {

    if (1 == C) {
      // one element of A per thread, just search for A into B
      const T searchVal = A[i];
      const size_t lb = pangolin::binary_search<Bounds::LOWER>(B, bSz, searchVal);
      if (lb < bSz) {
        threadCount += (B[lb] == searchVal ? 1 : 0);
      }
    } else {

      const T *chunkBegin = &A[i];
      const T *chunkEnd = &A[i + C];
      if (chunkEnd > &A[aSz]) {
        chunkEnd = &A[aSz];
      }

      // find the lower bound of the beginning of the A-chunk in B
      const size_t lb = pangolin::binary_search<Bounds::LOWER>(B, bSz, chunkBegin[0]);

      // Search for the A chunk in B, starting at the lower bound
      threadCount += pangolin::serial_sorted_count_linear(chunkBegin, chunkEnd, &B[lb], &B[bSz]);
    }
  }

  if (reduce) {
    // give lane 0 the total count discovered by the warp
    typedef cub::WarpReduce<uint64_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage tempStorage[WARPS_PER_BLOCK];
    uint64_t aggregate = WarpReduce(tempStorage[warpIdx]).Sum(threadCount);
    return aggregate;
  } else {
    return threadCount;
  }
}

/*! \brief threadblock cooperative count of elements in A that appear in B

    @param[out]   count           pointer to the count
    @param[in]    A               array A
    @param[in]    aSz             the number of elements in A
    @param[in]    B               array B
    @param[in]    bSz             the number of elements in B
    \tparam C           coarsening factor: elements per thread
    \tparam BLOCK_DIM_X Threadblock size

    Each thread takes a consective group of elements from A.
    The lower bound of the first element of that group into B is found with a binary search
    Then the search is executed in a linear fashion.
*/
template <size_t C, size_t BLOCK_DIM_X, typename T>
__device__ void block_sorted_count_binary(uint64_t *count, const T *const A, const size_t aSz, const T *const B,
                                          const size_t bSz) {

  static_assert(C != 0, "expect at least 1 element per thread");
  uint64_t threadCount = 0;

  // cover entirety of A with block
  for (size_t i = threadIdx.x * C; i < aSz; i += BLOCK_DIM_X * C) {

    const T *aChunkBegin = &A[i];
    const T *aChunkEnd = &A[i + C];
    if (aChunkEnd > &A[aSz]) {
      aChunkEnd = &A[aSz];
    }

    // find the lower bound of the beginning of the A-chunk in B
    ulonglong2 uu = pangolin::serial_sorted_search_binary(B, 0, bSz, *aChunkBegin);
    T lowerBound = uu.y;

    // Search for the A chunk in B, starting at the lower bound
    threadCount += pangolin::serial_sorted_count_linear(aChunkBegin, aChunkEnd, &B[lowerBound], &B[bSz]);
  }

  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

  if (threadIdx.x == 0) {
    *count = aggregate;
  }
}

/*! \brief threadblock cooperative count of elements in A that appear in B

    \tparam C           coarsening factor: elements per thread
    \tparam BLOCK_DIM_X Threadblock size
    \returns each thread's contribution to the count

    Each thread takes a consective group of elements from A.
    The lower bound of the first element of that group into B is found with a binary search
    Then the search is executed in a linear fashion.
*/
template <size_t C, size_t BLOCK_DIM_X, typename T>
__device__ uint64_t block_sorted_count_binary(const T *const A, //!< [in] array A
                                              const size_t aSz, //!< [in] the number of elements in A
                                              const T *const B, //!< [in] array B
                                              const size_t bSz  //!< [in] the number of elements in B
) {

  static_assert(C != 0, "expect at least 1 element per thread");
  uint64_t threadCount = 0;

  // cover entirety of A with block
  for (size_t i = threadIdx.x * C; i < aSz; i += BLOCK_DIM_X * C) {

    if (1 == C) {
      // one element of A per thread, just search for A into B
      const T searchVal = A[i];
      threadCount += serial_sorted_count_binary(B, 0, bSz, searchVal);
    } else {
      const T *aChunkBegin = &A[i];
      const T *aChunkEnd = &A[i + C];
      if (aChunkEnd > &A[aSz]) {
        aChunkEnd = &A[aSz];
      }

      // find the lower bound of the beginning of the A-chunk in B
      ulonglong2 uu = pangolin::serial_sorted_search_binary(B, 0, bSz, *aChunkBegin);
      T lowerBound = uu.y;

      // Search for the A chunk in B, starting at the lower bound
      threadCount += pangolin::serial_sorted_count_linear(aChunkBegin, aChunkEnd, &B[lowerBound], &B[bSz]);
    }
  }

  return threadCount;
}

} // namespace pangolin