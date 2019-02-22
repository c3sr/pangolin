#include "pangolin/logger.hpp"
#include "pangolin/par_graph.hpp"
#include "pangolin/reader/gc_tsv_reader.hpp"
#include "pangolin/utilities.hpp"
#include "pangolin/triangle_counter/vertex_tc.hpp"

#include <cub/cub.cuh>

PANGOLIN_BEGIN_NAMESPACE()

// count intersections between sorted lists a and b
__device__ static size_t linear_intersection_count(const Uint *const aBegin,
                                                   const Uint *const aEnd,
                                                   const Uint *const bBegin,
                                                   const Uint *const bEnd) {
  size_t count = 0;
  const auto *ap = aBegin;
  const auto *bp = bBegin;

  while (ap < aEnd && bp < bEnd) {

    if (*ap == *bp) {
      ++count;
      ++ap;
      ++bp;
    } else if (*ap < *bp) {
      ++ap;
    } else {
      ++bp;
    }
  }
  return count;
}

template <size_t BLOCK_DIM_X>
__global__ static void
kernel_linear(uint64_t *__restrict__ triangleCounts, // per block triangle count
              const Uint *rowStarts, const Uint *nonZeros,
              const char *isLocalNonZero, const size_t numRows) {

  // Specialize BlockReduce for a 1D block
  typedef cub::BlockReduce<size_t, BLOCK_DIM_X> BlockReduce;
  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;

  size_t count = 0;
  // one row per block
  for (Uint row = blockIdx.x; row < numRows; row += gridDim.x) {

    // offsets for head of edge
    const Uint head = row;

    // offsets for tail of edge
    const Uint tailStart = rowStarts[head];
    const Uint tailEnd = rowStarts[head + 1];

    // one thread per edge
    for (Uint tailOff = tailStart + threadIdx.x; tailOff < tailEnd;
         tailOff += BLOCK_DIM_X) {

      // only count local edges
      if (!isLocalNonZero || isLocalNonZero[tailOff]) {

        // edges from the head
        const Uint headEdgeStart = tailStart;
        const Uint headEdgeEnd = tailEnd;

        // edge from the tail
        const Uint tail = nonZeros[tailOff];
        const Uint tailEdgeStart = rowStarts[tail];
        const Uint tailEdgeEnd = rowStarts[tail + 1];

        count += linear_intersection_count(
            &nonZeros[headEdgeStart], &nonZeros[headEdgeEnd],
            &nonZeros[tailEdgeStart], &nonZeros[tailEdgeEnd]);
      }
    }
  }

  size_t aggregate = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) {
    triangleCounts[blockIdx.x] = aggregate;
  }
}

template <size_t BLOCK_DIM_X>
__global__ static void kernel_linear_shared(
    uint64_t *__restrict__ triangleCounts, // per block triangle count
    const Uint *rowStarts, const Uint *nonZeros, const char *isLocalNonZero,
    const size_t numRows) {

  // Specialize BlockReduce for a 1D block
  typedef cub::BlockReduce<size_t, BLOCK_DIM_X> BlockReduce;
  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ Uint headDsts[1024];

  size_t count = 0;
  // one row per block
  // all threads in the block share the same output edges from the head
  for (Uint row = blockIdx.x; row < numRows; row += gridDim.x) {

    // offsets for head of edge
    const Uint head = row;

    // offsets for tail of edge
    const Uint tailStart = rowStarts[head];
    const Uint tailEnd = rowStarts[head + 1];

    // pointers to beginning and end of row's nonzero columns
    const Uint *headEdgeBegin, *headEdgeEnd;

    // collaboratively load edges from head if they fit in shared memory
    if (tailEnd - tailStart < 1024) {
      for (size_t i = threadIdx.x + tailStart; i < tailEnd; i += BLOCK_DIM_X) {
        headDsts[i - tailStart] = nonZeros[i];
      }
      __syncthreads();
      headEdgeBegin = &headDsts[0];
      headEdgeEnd = &headDsts[tailEnd - tailStart];
    } else {
      headEdgeBegin = &nonZeros[tailStart];
      headEdgeEnd = &nonZeros[tailEnd];
    }

    // one thread per edge
    for (Uint tailOff = tailStart + threadIdx.x; tailOff < tailEnd;
         tailOff += BLOCK_DIM_X) {

      // only count local edges
      if (!isLocalNonZero || isLocalNonZero[tailOff]) {

        // edge from the tail
        const Uint tail = nonZeros[tailOff];
        const Uint tailEdgeStart = rowStarts[tail];
        const Uint tailEdgeEnd = rowStarts[tail + 1];

        count += linear_intersection_count(headEdgeBegin, headEdgeEnd,
                                           &nonZeros[tailEdgeStart],
                                           &nonZeros[tailEdgeEnd]);
      }
    }
  }

  size_t aggregate = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) {
    triangleCounts[blockIdx.x] = aggregate;
  }
}

// return 1 if search_val is in array between offets left and right, inclusive
__device__ static bool binary_search(const Uint *const array, size_t left,
                                     size_t right, const Uint search_val) {
  while (left <= right) {
    size_t mid = (left + right) / 2;
    Uint val = array[mid];
    if (val < search_val) {
      left = mid + 1;
    } else if (val > search_val) {
      right = mid - 1;
    } else { // val == search_val
      return 1;
    }
  }
  return 0;
}

template <size_t BLOCK_DIM_X>
__global__ static void
kernel_binary(uint64_t *__restrict__ triangleCounts, // per block triangle count
              const Uint *rowStarts, const Uint *nonZeros,
              const char *isLocalNonZero, const size_t numRows) {
  const size_t WARPS_PER_BLOCK = BLOCK_DIM_X / 32;
  static_assert(BLOCK_DIM_X % 32 ==
                0, "expect integer number of warps per block");

  const int warpIdx = threadIdx.x / 32; // which warp in thread block
  const int laneIdx = threadIdx.x % 32; // which thread in warp

  // Specialize BlockReduce for a 1D block
  typedef cub::BlockReduce<size_t, BLOCK_DIM_X> BlockReduce;
  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;

  size_t count = 0;
  // one row per block
  for (Int row = blockIdx.x; row < numRows; row += gridDim.x) {

    // offsets for head of edge
    const Int head = row;

    // offsets for tail of edge
    const Int tailStart = rowStarts[head];
    const Int tailEnd = rowStarts[head + 1];

    // one warp per edge
    for (Int tailOff = tailStart + warpIdx; tailOff < tailEnd;
         tailOff += WARPS_PER_BLOCK) {

      // only count local edges
      if (!isLocalNonZero || isLocalNonZero[tailOff]) {

        // edges from the head
        const Uint headEdgeStart = tailStart;
        const Uint headEdgeEnd = tailEnd;

        // edge from the tail
        const Uint tail = nonZeros[tailOff];
        const Uint tailEdgeStart = rowStarts[tail];
        const Uint tailEdgeEnd = rowStarts[tail + 1];

        // warp in parallel across shorter list to binary-search longer list
        if (headEdgeEnd - headEdgeStart < tailEdgeEnd - tailEdgeStart) {
          for (const Uint *u = &nonZeros[headEdgeStart] + laneIdx;
               u < &nonZeros[headEdgeEnd]; u += 32) {
            count +=
                binary_search(nonZeros, tailEdgeStart, tailEdgeEnd - 1, *u);
          }
        } else {
          for (const Uint *u = &nonZeros[tailEdgeStart] + laneIdx;
               u < &nonZeros[tailEdgeEnd]; u += 32) {
            count +=
                binary_search(nonZeros, headEdgeStart, headEdgeEnd - 1, *u);
          }
        }
      }
    }
  }

  size_t aggregate = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) {
    triangleCounts[blockIdx.x] = aggregate;
  }
}

VertexTC::VertexTC(Config &c) : CUDATriangleCounter(c) {

  std::string kernel = c.kernel_;

  if (kernel == "") {
    LOG(warn, "VertexTC defaulting to \"linear\" kernel");
    kernel = "linear";
  }

  if (kernel == "linear") {
    kernel_ = Kernel::LINEAR;
  } else if (kernel == "linear_shared") {
    kernel_ = Kernel::LINEAR_SHARED;
  } else if (kernel == "binary") {
    kernel_ = Kernel::BINARY;
  } else if (kernel == "hash") {
    kernel_ = Kernel::HASH;
  } else {
    LOG(critical , "Unknown triangle counting kernel \"{}\" for VertexTC",
        c.kernel_);
    exit(-1);
  }

  triangleCounts_d_ = std::vector<uint64_t *>(gpus_.size(), nullptr);
  rowOffsets_d_.resize(gpus_.size());
  nonZeros_d_.resize(gpus_.size());
  isLocalNonZero_d_.resize(gpus_.size());
}

VertexTC::~VertexTC() {}

void VertexTC::read_data(const std::string &path) {

  LOG(info, "reading {}", path);

  pangolin::GraphChallengeTSVReader reader(path);

  auto edgeList = reader.read_edges();

  // convert edge list to DAG by src < dst
  EdgeList filtered;
  for (const auto &e : edgeList) {
    if (e.first < e.second) {
      filtered.push_back(e);
    }
  }

  SPDLOG_TRACE(logger::console, "filtered edge list has {} entries", filtered.size());

  SPDLOG_DEBUG(logger::console, "building DAG");
  // for singe dag, no remote edges
  auto graph = UnifiedMemoryCSR::from_sorted_edgelist(filtered);

  numEdges_ = graph.nnz();
  numNodes_ = graph.num_rows();
  LOG(info, "{} edges", numEdges_);
  LOG(info, "{} nodes", numNodes_);
  LOG(info, "~{} KB storage", graph.bytes() / 1024);

  if (gpus_.size() == 1) {
    graphs_.push_back(graph);
  } else {
    graphs_ = graph.partition_nonzeros(gpus_.size());
  }

  if (graphs_.size() > 1) {
    size_t partNodes = 0;
    size_t partEdges = 0;
    size_t partSz = 0;
    for (const auto &graph : graphs_) {
      partNodes += graph.num_rows();
      partEdges += graph.nnz();
      partSz += graph.bytes();
    }
    LOG(info, "node replication {}", double(partNodes) / graph.num_rows());
    LOG(info, "edge replication {}", double(partEdges) / graph.nnz());
    LOG(info, "storage replication {}", double(partSz) / graph.bytes());
  }
}

void VertexTC::setup_data() {
  assert(gpus_.size());
  assert(graphs_.size() == gpus_.size());

  // one set of triangle counts per partition
  triangleCounts_.resize(gpus_.size());

  for (size_t i = 0; i < graphs_.size(); ++i) {
    const auto &graph = graphs_[i];
    const int dev = gpus_[i];
    CUDA_RUNTIME(cudaSetDevice(dev));

    dimGrids_.push_back(graph.num_rows());
    const dim3 &dimGrid = dimGrids_[i];

    // create space for one triangle count per block

    triangleCounts_[i].resize(dimGrid.x, 0);

    // device pointers are directly the managed memory pointers
    triangleCounts_d_[i] = triangleCounts_[i].data();
    rowOffsets_d_[i] = graph.row_offsets();
    nonZeros_d_[i] = graph.cols();
    isLocalNonZero_d_[i] = graph.is_local_cols();
  }

  for (const auto i : gpus_) {
    SPDLOG_TRACE(logger::console, "synchronizing GPU {}", i);
    CUDA_RUNTIME(cudaSetDevice(i));
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }
}

size_t VertexTC::count() {
  assert(graphs_.size() == gpus_.size());
  for (size_t i = 0; i < graphs_.size(); ++i) {
    const auto &graph = graphs_[i];
    const int dev = gpus_[i];
    const dim3 &dimGrid = dimGrids_[i];
    const size_t numRows = graph.num_rows();

    switch (kernel_) {
    case Kernel::LINEAR: {
      SPDLOG_DEBUG(logger::console, "linear search kernel");
      const size_t BLOCK_DIM_X = 128;
      dim3 dimBlock(BLOCK_DIM_X);
      SPDLOG_DEBUG(logger::console, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
      kernel_linear<BLOCK_DIM_X><<<dimGrid, dimBlock>>>(
          triangleCounts_d_[i], rowOffsets_d_[i], nonZeros_d_[i],
          isLocalNonZero_d_[i], numRows);
      CUDA_RUNTIME(cudaGetLastError());
      break;
    }
    case Kernel::LINEAR_SHARED: {
      SPDLOG_DEBUG(logger::console, "linear_shared search kernel");
      const size_t BLOCK_DIM_X = 128;
      dim3 dimBlock(BLOCK_DIM_X);
      SPDLOG_DEBUG(logger::console, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
      kernel_linear_shared<BLOCK_DIM_X><<<dimGrid, dimBlock>>>(
          triangleCounts_d_[i], rowOffsets_d_[i], nonZeros_d_[i],
          isLocalNonZero_d_[i], numRows);
      CUDA_RUNTIME(cudaGetLastError());
      break;
    }
    case Kernel::BINARY: {
      SPDLOG_DEBUG(logger::console, "binary search kernel");
      const size_t BLOCK_DIM_X = 512;
      dim3 dimBlock(BLOCK_DIM_X);
      dim3 dimGrid(graph.num_rows());
      SPDLOG_DEBUG(logger::console, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
      kernel_binary<BLOCK_DIM_X><<<dimGrid, dimBlock>>>(
          triangleCounts_d_[i], rowOffsets_d_[i], nonZeros_d_[i],
          isLocalNonZero_d_[i], numRows);
      CUDA_RUNTIME(cudaGetLastError());
      break;
    }
    case Kernel::HASH: {
      LOG(critical , "hash kernel unimplmeneted");
      exit(-1);
      break;
    }
    default: {
      LOG(critical , "unexpected kernel type.");
      exit(-1);
    }
    }
  }

  Uint total = 0;

  for (size_t i = 0; i < graphs_.size(); ++i) {
    const int &dev = gpus_[i];
    const dim3 &dimGrid = dimGrids_[i];
    SPDLOG_DEBUG(logger::console, "waiting for GPU {}", dev);
    CUDA_RUNTIME(cudaSetDevice(dev));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    Uint partitionTotal = 0;
    SPDLOG_DEBUG(logger::console, "cpu reduction for GPU {}", dev);
    for (size_t j = 0; j < dimGrid.x; ++j) {
      partitionTotal += triangleCounts_[i][j];
    }
    SPDLOG_DEBUG(logger::console, "partition had {} triangles", partitionTotal);
    total += partitionTotal;
  }

  return total;
}

PANGOLIN_END_NAMESPACE()