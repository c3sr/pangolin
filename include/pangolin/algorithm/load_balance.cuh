#pragma once

#include <cub/cub.cuh>

#include "binary_search.cuh"
#include "pangolin/utilities.hpp"

namespace pangolin {

/*!

  Some number of objects, each of which produces `count` work-items.
  Upper-bound search of work item index into exclusive scan of work-item counts.

  \tparam OI the object index type
  \tparam WI the work-item index type

*/
template <typename OI, typename WI>
inline void load_balance(OI *indices,           //<! [out] the object index that produced each work item
                         const WI numWorkItems, //<! [in] the total number of work items
                         const WI *counts,      //<! [in] the number of work items produced by each object
                         const OI numObjects    //<! [in] the number of objects

) {
  WI wi = 0;
  OI oi = 0;
  WI exclScanCounts = 0;
  while (wi < numWorkItems || oi < numObjects) {
    bool pred;
    if (oi >= numObjects) { // all remaining work-items are from the last object
      pred = true;
    } else if (wi >= numWorkItems) { // all work items have been produced, final objects didn't contribute any
      pred = false;
    } else {
      pred = (wi < exclScanCounts);
    }
    if (pred) {
      indices[wi++] = oi - 1;
    } else {
      exclScanCounts += counts[oi];
      ++oi;
    }
  }
}

/*! A vectorized sorted search where one vector is just [0...numWorkItems)

  \tparam OI the object index type
  \tparam WI the work-item index type

  \return an array of length numWorkItems containing the object index that produced each work item
*/
template <typename OI, typename WI>
__global__ void grid_load_balance_kernel(
    OI *indices,              //<! [out] the object index that produced each work item
    const WI numWorkItems,    //<! [in] the total number of work items
    const WI *exclScanCounts, //<! [in] exclusive scan of the number of work items produced by each object
    const OI numObjects       //<! [in] the number of objects

) {
  for (WI wi = blockDim.x * blockIdx.x + threadIdx.x; wi < numWorkItems; wi += gridDim.x * blockDim.x) {
    indices[wi] = binary_search<Bounds::UPPER>(exclScanCounts, numObjects, wi) - 1;
  }
}

/*! An exclusive scan followed by a grid_load_balance_kernel

  \tparam OI the object index type
  \tparam WI the work-item index type

  \return an array of length numWorkItems containing the object index that produced each work item
*/
template <typename OI, typename WI>
void device_load_balance(OI *indices,            //<! [out] the object index that produced each work item
                         const WI numWorkItems,  //<! [in] the total number of work items
                         const WI *counts,       //<! [in] the number of work items produced by each object
                         const OI numObjects,    //<! [in] the number of objects
                         cudaStream_t stream = 0 //<! [in] the stream to execute in
) {

  // allocate space for exclusive scan results
  size_t exclScanBytes = sizeof(WI) * numObjects;
  LOG(debug, "allocate {}B for exclusive scan output", exclScanBytes);
  WI *exclScanCounts = nullptr;
  CUDA_RUNTIME(cudaMalloc(&exclScanCounts, exclScanBytes));

  // compute temp storage needed for exclusive sum
  void *tempStorage = nullptr;
  size_t tempStorageBytes = 0;
  cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, counts, exclScanCounts, numObjects, stream);
  CUDA_RUNTIME(cudaGetLastError());

  // allocate temporary storage
  LOG(debug, "allocate {}B for exclusive scan temp storage", tempStorageBytes);
  CUDA_RUNTIME(cudaMalloc(&tempStorage, tempStorageBytes));

  // run exclusive scan
  LOG(debug, "launch exclusive scan");
  cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, counts, exclScanCounts, numObjects, stream);
  CUDA_RUNTIME(cudaGetLastError());

  // run load-balanced search
  LOG(debug, "launch grid_load_balance_kernel<<<{}, {}, 0, {}>>>", 512, 512, uintptr_t(stream));
  grid_load_balance_kernel<<<512, 512, 0, stream>>>(indices, numWorkItems, exclScanCounts, numObjects);
  CUDA_RUNTIME(cudaGetLastError());

  // free temporary storage
  LOG(debug, "free temporary storage");
  CUDA_RUNTIME(cudaFree(tempStorage));
}

} // namespace pangolin