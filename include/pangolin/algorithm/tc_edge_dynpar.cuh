#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

namespace pangolin {

/*! \brief count of elements in A that appear in B

    @param[inout] count       pointer to the count. caller should initialize to 0.
    \param        A           the array of needles
    \param        aSz         the number of elements in A
    \param        B           the haystack
    \param        bSz         the number of elements in B
    \tparam       BLOCK_DIM_X the dimension of the thread block
*/
template <size_t BLOCK_DIM_X, typename T>
__global__ void kernel_sorted_count_binary(uint64_t *count, const T *const A, const size_t aSz, const T *const B,
                                           const size_t bSz) {
  grid_sorted_count_binary<BLOCK_DIM_X>(count, A, aSz, B, bSz);
}

/*! \brief Count triangles in a CSR/COO hybrid matrix

    @param[out] edgeTriangleCounts array of number of non-zeros per-edge triangle counts
    @param[in]  csr                A CSR/COO matrix

    Produce a per-edge triangle count using CUDA dynamic parallelism
*/
template <typename CsrCooMatrix> __global__ void kernel_edgetc_dynpar(uint64_t *edgeTriangleCounts, CsrCooMatrix csr) {
  using Index = typename CsrCooMatrix::index_type;

  const int gx = blockDim.x * blockIdx.x + threadIdx.x;

  const uint64_t numEdges = csr.nnz();
  for (uint64_t i = gx; i < numEdges; i += blockDim.x * gridDim.x) {
    // printf("working on edge %lu\n", i);
    Index src = csr.device_row_ind()[i];
    Index dst = csr.device_col_ind()[i];
    Index srcStart = csr.device_row_ptr()[src];
    Index srcStop = csr.device_row_ptr()[src + 1];
    Index dstStart = csr.device_row_ptr()[dst];
    Index dstStop = csr.device_row_ptr()[dst + 1];
    // printf("%d [%d-%d] -> %d [%d-%d]\n", src, srcStart, srcStop, dst, dstStart, dstStop);

    const Index *srcBegin = &csr.device_col_ind()[srcStart];
    const size_t srcSz = srcStop - srcStart;
    const Index *dstBegin = &csr.device_col_ind()[dstStart];
    const size_t dstSz = dstStop - dstStart;

    constexpr size_t dimBlock = 256;
    uint64_t *edgeCount = &edgeTriangleCounts[i];
    if (srcSz > dstSz) { // src has more neighbors than dst, search for srcs in parallel
      const size_t dimGrid = (srcSz + dimBlock - 1) / dimBlock;
      kernel_sorted_count_binary<dimBlock><<<dimGrid, dimBlock>>>(edgeCount, srcBegin, srcSz, dstBegin, dstSz);
    } else {
      const size_t dimGrid = (dstSz + dimBlock - 1) / dimBlock;
      kernel_sorted_count_binary<dimBlock><<<dimGrid, dimBlock>>>(edgeCount, dstBegin, dstSz, srcBegin, srcSz);
    }
  }
}

/*! \brief Count triangles

  \tparam CsrCooMatrix a DAG in hybrid CSR/COO format
 */
template <typename CsrCooMatrix> uint64_t tc_edge_dynpar(const CsrCooMatrix &csr) {

  // allocate space for the total count
  uint64_t *count = nullptr;
  CUDA_RUNTIME(cudaMallocManaged(&count, sizeof(*count)));
  *count = 0;

  // allocate temporary storage of one count per edge
  Vector<uint64_t> edgeTriangleCounts(csr.nnz());

  // one thread per non-zero (edge)
  constexpr size_t dimBlock = 512;
  const size_t dimGrid = (csr.nnz() + dimBlock - 1) / dimBlock;
  // constexpr size_t dimBlock = 1;
  // const size_t dimGrid = 1;

  // count triangles on the device
  kernel_edgetc_dynpar<<<dimGrid, dimBlock>>>(edgeTriangleCounts.data(), csr.view());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  // reduction
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, edgeTriangleCounts.data(), count, csr.nnz());
  CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, edgeTriangleCounts.data(), count, csr.nnz());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  CUDA_RUNTIME(cudaFree(d_temp_storage));
  uint64_t ret = *count;
  CUDA_RUNTIME(cudaFree(count));

  return ret;
}

} // namespace pangolin