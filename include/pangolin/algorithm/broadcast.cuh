#pragma once

/*! Broadcast value from threadIdx.x root to all threads in the block
 */
template <typename T> T block_broadcast(const T val, const int root) {
  __shared__ T sharedVal;

  if (threadIdx.x == root) {
    sharedVal = val;
  }
  __syncthreads();
  return sharedVal;
}