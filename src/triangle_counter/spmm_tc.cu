
#include "pangolin/logger.hpp"
#include "pangolin/reader/edge_list_reader.hpp"
#include "pangolin/triangle_counter/spmm_tc.hpp"
#include "pangolin/utilities.hpp"

#include <cub/cub.cuh>
#include <limits>
#include <nvToolsExt.h>

PANGOLIN_BEGIN_NAMESPACE()

/*!
  return the number of common elements between sorted lists a and b
*/
template <typename U>
__device__ static uint64_t intersection_count(const U *ab, //!< beginning of a
                                              const U *ae, //!< end of a
                                              const U *bb, //!< beginning of b
                                              const U *be  //!< end of b
) {
  const U *a = ab;
  const U *b = bb;
  uint64_t count = 0;

  while (a < ae && b < be) {
    if (*a < *b) {
      ++a;
    } else if (*b < *a) {
      ++b;
    } else {
      ++a;
      ++b;
      ++count;
    }
  }
  return count;
}

/*! Count triangles using LxU*L CSR method

One thread per row
*/
template <typename Index>
__global__ void
spmm_csr_csr(Index *edgeSrc,     //!< src of edge (at least # edges long)
             Index *edgeDst,     //!< dst of edge (at least # edges long)
             uint64_t *edgeCnt,  //!< tri count of edge (at least # edges long)
             uint64_t *nextEdge, //!< pointer to next available edge
             const pangolin::GPUCSRView<Index> aL, //!< aL*aU, lower-triangular
             const pangolin::GPUCSRView<Index> aU  //!< aL*aU, upper-triagular
) {
  static_assert(sizeof(long long unsigned) == sizeof(uint64_t), "");
  const size_t num_rows = aL.num_rows();

  // each thread handles a row
  for (Index row = blockDim.x * blockIdx.x + threadIdx.x; row < num_rows;
       row += blockDim.x * gridDim.x) {
    // printf("working on src %lu\n", row);
    const Index row_start = aL.rowOffset_[row];
    const Index row_end = aL.rowOffset_[row + 1];
    // produce a result for non-zero columns in that row
    for (int colIdx = row_start; colIdx < row_end; colIdx++) {
      const Index col = aL.col_[colIdx];
      // printf("working on e %u-%u\n", row, col);
      // look up the start and end offsets of the non-zero row entries in aU
      const Index dst_start = aU.rowOffset_[col];
      const Index dst_end = aU.rowOffset_[col + 1];
      // printf("src neighbors are %u-%u\n", row_start, row_end);
      for (Index src = row_start; src < row_end; ++src) {
        // printf("%u ", aL.col_[src]);
      }
      // printf("\n");
      // printf("dst neighbors are %u-%u\n", dst_start, dst_end);
      for (Index dst = dst_start; dst < dst_end; ++dst) {
        // printf("%u ", aU.col_[dst]);
      }
      // printf("\n");
      // find the number of equal non-zero elements in the rows and column
      uint64_t dot = intersection_count(&aL.col_[row_start], &aL.col_[row_end],
                                        &aU.col_[dst_start], &aU.col_[dst_end]);
      // printf("dot is %lu\n", dot);
      // save triangle count for edge
      uint64_t edgeIdx =
          atomicAdd(reinterpret_cast<long long unsigned *>(nextEdge), 1ull);
      edgeSrc[edgeIdx] = row;
      edgeDst[edgeIdx] = col;
      edgeCnt[edgeIdx] = dot;
    }
  }
}

SpmmTC::SpmmTC(Config &c) : CUDATriangleCounter(c) {
  nvtxRangePush(__PRETTY_FUNCTION__);
  LOG(info, "Sparse Matrix-Matrix Multiply TC, sizeof(Int) = {}", sizeof(Int));
  nvtxRangePop();
}

SpmmTC::~SpmmTC() {
  nvtxRangePush(__PRETTY_FUNCTION__);
  CUDA_RUNTIME(cudaFree(edgeSrc_));
  CUDA_RUNTIME(cudaFree(edgeDst_));
  CUDA_RUNTIME(cudaFree(edgeCnt_));
  CUDA_RUNTIME(cudaFree(nextEdge_));
  nvtxRangePop();
}

void SpmmTC::read_data(const std::string &path) {
  nvtxRangePush(__PRETTY_FUNCTION__);
  LOG(info, "reading {}", path);
  auto *reader = pangolin::EdgeListReader::from_file(path);
  auto edgeList = reader->read_all();
  if (edgeList.size() == 0) {
    LOG(warn, "empty edge list");
  }
  SPDLOG_DEBUG(logger::console, "building GPUCSR");
  aL_ = std::move(
      pangolin::GPUCSR<Uint>::from_edgelist(edgeList, [](const Edge &e) {
        return e.first <= e.second;
      })); // keep src > dst
  aU_ = std::move(
      pangolin::GPUCSR<Uint>::from_edgelist(edgeList, [](const Edge &e) {
        return e.first >= e.second;
      })); // keep src < dst

  // LOG(info, "Lower-triangular: {} nodes", aL_.num_nodes());
  LOG(info, "Lower-triangular: {} edges", aL_.nnz());
  // LOG(info, "Upper-triangular: {} nodes", aU_.num_nodes());
  LOG(info, "Upper-triangular: {} edges", aU_.nnz());
  nvtxRangePop();
}

void SpmmTC::setup_data() {
  nvtxRangePush(__PRETTY_FUNCTION__);

  const size_t edgeCount = aL_.nnz();
  const size_t edgeSrcBytes = edgeCount * sizeof(*edgeSrc_);
  const size_t edgeDstBytes = edgeCount * sizeof(*edgeDst_);
  const size_t edgeCntBytes = edgeCount * sizeof(*edgeCnt_);
  SPDLOG_DEBUG(logger::console, "allocating {}B for edge sources",
               edgeSrcBytes);
  CUDA_RUNTIME(cudaMallocManaged(&edgeSrc_, edgeSrcBytes));
  SPDLOG_DEBUG(logger::console, "allocating {}B for edge destinations",
               edgeDstBytes);
  CUDA_RUNTIME(cudaMallocManaged(&edgeDst_, edgeDstBytes));
  SPDLOG_DEBUG(logger::console, "allocating {}B for edge triangle counts",
               edgeCntBytes);
  CUDA_RUNTIME(cudaMallocManaged(&edgeCnt_, edgeCntBytes));
  SPDLOG_DEBUG(logger::console, "allocating {}B for next edge offset counter",
               sizeof(*nextEdge_));
  CUDA_RUNTIME(cudaMallocManaged(&nextEdge_, sizeof(*nextEdge_)));
  nvtxRangePop();
}

size_t SpmmTC::count() {
  nvtxRangePush(__PRETTY_FUNCTION__);

  const size_t numDev = gpus_.size();
  if (numDev > 1) {
    LOG(warn, "SpmmTC uses only 1 gpu, using {}", gpus_[0]);
  }

  CUDA_RUNTIME(cudaSetDevice(gpus_[0]));

  dim3 dimBlock(256);
  size_t desiredGridSize = (aL_.nnz() + dimBlock.x - 1) / dimBlock.x;
  dim3 dimGrid(
      std::min(size_t(std::numeric_limits<int>::max()), desiredGridSize));
  // dim3 dimBlock(1);
  // dim3 dimGrid(1);
  SPDLOG_DEBUG(logger::console, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
  spmm_csr_csr<<<dimGrid, dimBlock>>>(edgeSrc_, edgeDst_, edgeCnt_, nextEdge_,
                                      aL_.view(), aU_.view());
  SPDLOG_DEBUG(logger::console, "launched kernel");
  CUDA_RUNTIME(cudaGetLastError());

  for (int i : std::set<int>(gpus_.begin(), gpus_.end())) {
    CUDA_RUNTIME(cudaSetDevice(i));
    SPDLOG_DEBUG(logger::console, "Waiting for GPU {}", i);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  nvtxRangePush("final reduction");
  size_t final_total = 0;
  auto start = std::chrono::system_clock::now();
  if (1) // CPU
  {
    SPDLOG_DEBUG(logger::console, "CPU reduction");
    size_t total = 0;
    for (size_t i = 0; i < aL_.nnz(); ++i) {
      total += edgeCnt_[i];
    }
    final_total = total;
  } else { // GPU
    SPDLOG_DEBUG(logger::console, "GPU reduction");
    size_t *total;
    CUDA_RUNTIME(cudaMallocManaged(&total, sizeof(*total)));
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, edgeCnt_, total,
                           aL_.nnz());
    // Allocate temporary storage
    SPDLOG_DEBUG(logger::console, "{}B for cub::DeviceReduce::Sum temp storage",
                 temp_storage_bytes);
    CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, edgeCnt_, total,
                           aL_.nnz());
    CUDA_RUNTIME(cudaDeviceSynchronize());
    final_total = *total;
    CUDA_RUNTIME(cudaFree(total));
    CUDA_RUNTIME(cudaFree(d_temp_storage));
  }
  auto elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  nvtxRangePop(); // final reduction
  SPDLOG_DEBUG(logger::console, "Final reduction {}s", elapsed);

  nvtxRangePop();
  return final_total;
}

PANGOLIN_END_NAMESPACE()