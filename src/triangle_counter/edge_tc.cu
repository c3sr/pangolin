#include "graph/logger.hpp"
#include "graph/par_graph.hpp"
#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/utilities.hpp"
#include "graph/triangle_counter/edge_tc.hpp"

#include <cub/cub.cuh>

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
kernel_linear(uint64_t *__restrict__ triangleCounts, // per edge triangle count
              const Uint *rowStarts, const Uint *edgeSrc, const Uint *nonZeros,
              const char *isLocalNonZero, const size_t numEdges) {


  const size_t gIdx = threadIdx.x + blockIdx.x * BLOCK_DIM_X;

  // one thread per edge
  for (Uint edgeIdx = gIdx; edgeIdx < numEdges; edgeIdx += BLOCK_DIM_X * gridDim.x) {

    // only count local edges
    if (!isLocalNonZero || isLocalNonZero[edgeIdx]) {

        size_t count = 0;

        // head and tail of edge
        const Uint head = edgeSrc[edgeIdx];
        const Uint tail = nonZeros[edgeIdx];

        // neighbor offsets for head of edge
        const Uint headOffStart = rowStarts[head];
        const Uint headOffEnd = rowStarts[head + 1];

        // neighbor offsets for tail of edge
        const Uint tailOffStart = rowStarts[tail];
        const Uint tailOffEnd = rowStarts[tail + 1];



        count += linear_intersection_count(
            &nonZeros[headOffStart], &nonZeros[headOffEnd],
            &nonZeros[tailOffStart], &nonZeros[tailOffEnd]);
            
            triangleCounts[edgeIdx] = count;
    }

  }
}


#if 0

template <size_t BLOCK_DIM_X>
__global__ static void
kernel_linear(uint64_t *__restrict__ triangleCounts, // per block triangle count
              const Uint *rowStarts, const Uint *edgeSrc, const Uint *nonZeros,
              const char *isLocalNonZero, const size_t numEdges) {

  typedef cub::WarpReduce<size_t> WarpReduce;
  constexpr int WARPS_PER_BLOCK = BLOCK_DIM_X / 32;
  static_assert(BLOCK_DIM_X % 32 == 0, "need integer number of warps per block");
  __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];

  const int warpIdx = threadIdx.x / 32; // which warp in thread block
  const int laneIdx = threadIdx.x % 32; // which thread in warp

  // one warp per edge
  for (Uint edgeIdx = warpIdx + blockIdx.x * WARPS_PER_BLOCK; edgeIdx < numEdges; edgeIdx += WARPS_PER_BLOCK * gridDim.x) {

    size_t count = 0;

    // head and tail of edge
    const Uint head = edgeSrc[edgeIdx];
    const Uint tail = nonZeros[edgeIdx];

    // neighbor offsets for head of edge
    const Uint headOffStart = rowStarts[head];
    const Uint HeadOffEnd = rowStarts[head + 1];

    // neighbor offsets for tail of edge
    const Uint tailOffStart = rowStarts[tail];
    const Uint tailOffEnd = rowStarts[tail + 1];

    // one thread per neighbor
    for (Uint tailOff = tailStart + laneIdx; tailOff < tailEnd;
         tailOff += 32) {

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

    size_t aggregate = WarpReduce(temp_storage[warpIdx]).Sum(count);
    if (laneIdx == 0) {
        triangleCounts[edgeIdx] = aggregate;
    }
  }
}

#endif

#if 0
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
#endif

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

#if 0
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
#endif


EdgeTC::EdgeTC(Config &c) : CUDATriangleCounter(c) {

  std::string kernel = c.kernel_;

  if (kernel == "") {
    LOG(warn, "EdgeTC defaulting to \"linear\" kernel");
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
    LOG(critical, "Unknown triangle counting kernel \"{}\" for EdgeTC",
        c.kernel_);
    exit(-1);
  }

  triangleCounts_d_ = std::vector<uint64_t *>(gpus_.size(), nullptr);
  rowOffsets_d_.resize(gpus_.size());
  cols_d_.resize(gpus_.size());
  isLocalCol_d_.resize(gpus_.size());
}

EdgeTC::~EdgeTC() {}

void EdgeTC::read_data(const std::string &path) {

  LOG(info, "reading {}", path);

  GraphChallengeTSVReader reader(path);

  auto edgeList = reader.read_edges();

  // convert edge list to DAG by src < dst
  EdgeList filtered;
  for (const auto &e : edgeList) {
    if (e.first < e.second) {
      filtered.push_back(e);
    }
  }

  LOG(trace, "filtered edge list has {} entries", filtered.size());

  LOG(debug, "building DAG");
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


  // build source edge lists for each graph
  for (const auto &g : graphs_) {
      CUDAManagedVector<Uint> edgeSrc;
      for (size_t head= 0; head < g.num_rows(); ++head) {
          const size_t tailOffBegin = g.row_offsets()[head];
          const size_t tailOffEnd = g.row_offsets()[head+1];
          for (size_t tailOff = tailOffBegin; tailOff < tailOffEnd; ++tailOff) {
              edgeSrc.push_back(static_cast<Uint>(head));
          }
      }
      edgeSrc_.push_back(edgeSrc);
      LOG(debug, "created src edge list of length {}", edgeSrc.size());
      assert(edgeSrc.size() == g.nnz());
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

void EdgeTC::setup_data() {
  assert(gpus_.size());
  assert(graphs_.size() == gpus_.size());
  assert(edgeSrc_.size() == gpus_.size());

  // one set of triangle counts per partition
  triangleCounts_.resize(gpus_.size());

  // one list of edge srcs per partitions
  rows_d_.resize(gpus_.size());

  for (size_t i = 0; i < graphs_.size(); ++i) {
    const auto &graph = graphs_[i];
    const int dev = gpus_[i];
    CUDA_RUNTIME(cudaSetDevice(dev));

    // create space for one triangle count per edge
    triangleCounts_[i].resize(graph.nnz(), 0);

    // device pointers are directly the managed memory pointers
    triangleCounts_d_[i] = triangleCounts_[i].data();
    rowOffsets_d_[i] = graph.row_offsets();
    cols_d_[i] = graph.cols();
    rows_d_[i] = edgeSrc_[i].data();
    isLocalCol_d_[i] = graph.is_local_cols();
  }

  for (const auto i : gpus_) {
    LOG(trace, "synchronizing GPU {}", i);
    CUDA_RUNTIME(cudaSetDevice(i));
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }
}

size_t EdgeTC::count() {
  assert(graphs_.size() == gpus_.size());
  for (size_t i = 0; i < graphs_.size(); ++i) {
    const auto &graph = graphs_[i];
    const int dev = gpus_[i];
    const size_t numEdges = graph.nnz();

    switch (kernel_) {
    case Kernel::LINEAR: {
      LOG(debug, "linear search kernel");
      const size_t BLOCK_DIM_X = 512;
      const dim3 dimBlock(BLOCK_DIM_X);
      const size_t warpsPerBlock = dimBlock.x / 32;
      const size_t numGridWarps = graph.nnz();
      dim3 dimGrid((numGridWarps + warpsPerBlock - 1) / warpsPerBlock);
      dimGrid.x = std::min(dimGrid.x, static_cast<typeof(dimGrid.x)>((((uint64_t) 1) << 31) - 1)); // 2^32 - 1
      LOG(debug, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
      kernel_linear<BLOCK_DIM_X><<<dimGrid, dimBlock>>>(
          triangleCounts_d_[i], rowOffsets_d_[i], rows_d_[i], cols_d_[i],
          isLocalCol_d_[i], numEdges);
      CUDA_RUNTIME(cudaGetLastError());
      break;
    }
    case Kernel::LINEAR_SHARED: {
        LOG(critical, "hash kernel unimplmeneted");
        exit(-1);
        break;
    }
    case Kernel::BINARY: {
        LOG(critical, "hash kernel unimplmeneted");
        exit(-1);
        break;
    }
    case Kernel::HASH: {
      LOG(critical, "hash kernel unimplmeneted");
      exit(-1);
      break;
    }
    default: {
      LOG(critical, "unexpected kernel type.");
      exit(-1);
    }
    }
  }

  Uint total = 0;

  for (size_t i = 0; i < graphs_.size(); ++i) {
    const auto &graph = graphs_[i];
    const int &dev = gpus_[i];
    LOG(debug, "waiting for GPU {}", dev);
    CUDA_RUNTIME(cudaSetDevice(dev));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    Uint partitionTotal = 0;
    LOG(debug, "cpu reduction for GPU {}", dev);
    for (size_t j = 0; j < graph.nnz(); ++j) {
      partitionTotal += triangleCounts_[i][j];
    }
    LOG(debug, "partition had {} triangles", partitionTotal);
    total += partitionTotal;
  }

  return total;
}
