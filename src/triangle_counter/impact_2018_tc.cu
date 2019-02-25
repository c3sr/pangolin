#include "pangolin/logger.hpp"
#include "pangolin/reader/gc_tsv_reader.hpp"
#include "pangolin/triangle_counter/impact_2018_tc.hpp"
#include "pangolin/utilities.hpp"

#include <limits>
#include <nvToolsExt.h>

namespace pangolin {

__global__ static void kernel_tc(size_t *triangleCounts, Int *edgeSrc,
                                 Int *edgeDst, Int *nodes, size_t edgeOffset,
                                 size_t numEdges) {

  const Int gx = blockIdx.x * blockDim.x + threadIdx.x;

  for (Int i = gx + edgeOffset; i < edgeOffset + numEdges;
       i += blockDim.x * gridDim.x) {

    // get the src and dst node for this edge
    const Int src = edgeSrc[i];
    const Int dst = edgeDst[i];

    const Int srcEdge = nodes[src];
    const Int srcEdgeEnd = nodes[src + 1];

    const Int dstEdge = nodes[dst];
    const Int dstEdgeEng = nodes[dst + 1];

    size_t count = 0;

    Int *srcPtr = &edgeDst[srcEdge];
    Int *dstPtr = &edgeDst[dstEdge];

    while (srcPtr < &edgeDst[srcEdgeEnd] && dstPtr < &edgeDst[dstEdgeEng]) {

      const Int srcNbr = *srcPtr; // neighbor vertex of edge src
      const Int dstNbr = *dstPtr; // neighbor vertex of edge dst

      if (srcNbr == dstNbr) {
        ++count;
        ++srcPtr;
        ++dstPtr;
      } else if (srcNbr < dstNbr) {
        ++srcPtr;
      } else {
        ++dstPtr;
      }
    }

    triangleCounts[i] = count;
  }
}

IMPACT2018TC::IMPACT2018TC(Config &c) : CUDATriangleCounter(c) {
  nvtxRangePush(__PRETTY_FUNCTION__);
  SPDLOG_DEBUG(logger::console, "IMPACT 2018 TC, sizeof(Int) = {}",
               sizeof(Int));

  if (c.storage_ == "um") {
    GPUMemoryKind_ = GPUMemoryKind::Unified;
  } else if (c.storage_ == "zc") {
    GPUMemoryKind_ = GPUMemoryKind::ZeroCopy;
  } else {
    LOG(critical, "unknown gpu storage kind \"{}\"", c.storage_);
    exit(-1);
  }

  unifiedMemoryHints_ = c.hints_;

  nvtxRangePop();
}

IMPACT2018TC::~IMPACT2018TC() {
  nvtxRangePush(__PRETTY_FUNCTION__);
  switch (GPUMemoryKind_) {
  case GPUMemoryKind::Unified: {
    CUDA_RUNTIME(cudaFree(edgeSrc_d_));
    CUDA_RUNTIME(cudaFree(edgeDst_d_));
    CUDA_RUNTIME(cudaFree(nodes_d_));
    CUDA_RUNTIME(cudaFree(triangleCounts_));
    break;
  }
  case GPUMemoryKind::ZeroCopy: {
    CUDA_RUNTIME(cudaHostUnregister(hostDAG_.edgeSrc_.data()));
    CUDA_RUNTIME(cudaHostUnregister(hostDAG_.edgeDst_.data()));
    CUDA_RUNTIME(cudaHostUnregister(hostDAG_.nodes_.data()));
    CUDA_RUNTIME(cudaFreeHost(triangleCounts_));
    break;
  }
  default:
    LOG(error, "unexpected GPUMemoryKind in dtor");
  }
  nvtxRangePop();
}

void IMPACT2018TC::read_data(const std::string &path) {
  nvtxRangePush(__PRETTY_FUNCTION__);
  LOG(info, "reading {}", path);
  pangolin::GraphChallengeTSVReader reader(path);
  auto edgeList = reader.read_edges();
  SPDLOG_DEBUG(logger::console, "building DAG");
  hostDAG_ = DAG2019::from_edgelist(edgeList);

  LOG(info, "{} nodes", hostDAG_.num_nodes());
  LOG(info, "{} edges", hostDAG_.num_edges());
  nvtxRangePop();
}

void IMPACT2018TC::setup_data() {

  nvtxRangePush(__PRETTY_FUNCTION__);
  const size_t edgeBytes = hostDAG_.edgeSrc_.size() * sizeof(Int);
  const size_t nodeBytes = hostDAG_.nodes_.size() * sizeof(Int);
  const size_t countBytes = hostDAG_.num_edges() * sizeof(*triangleCounts_);

  switch (GPUMemoryKind_) {
  case GPUMemoryKind::Unified: {
    CUDA_RUNTIME(cudaMallocManaged(&edgeSrc_d_, edgeBytes));
    CUDA_RUNTIME(cudaMallocManaged(&edgeDst_d_, edgeBytes));
    CUDA_RUNTIME(cudaMallocManaged(&nodes_d_, nodeBytes));
    CUDA_RUNTIME(cudaMallocManaged(&triangleCounts_, countBytes));

    SPDLOG_DEBUG(logger::console, "copying to unified memory");
    CUDA_RUNTIME(cudaMemcpy(edgeSrc_d_, hostDAG_.edgeSrc_.data(), edgeBytes,
                            cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpy(edgeDst_d_, hostDAG_.edgeDst_.data(), edgeBytes,
                            cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpy(nodes_d_, hostDAG_.nodes_.data(), nodeBytes,
                            cudaMemcpyDefault));

    // processor id is ignored
    if (unifiedMemoryHints_) {
      LOG(info, "using unified memory hints");
      CUDA_RUNTIME(
          cudaMemAdvise(edgeSrc_d_, edgeBytes, cudaMemAdviseSetReadMostly, 0));
      CUDA_RUNTIME(
          cudaMemAdvise(edgeDst_d_, edgeBytes, cudaMemAdviseSetReadMostly, 0));
      CUDA_RUNTIME(
          cudaMemAdvise(nodes_d_, nodeBytes, cudaMemAdviseSetReadMostly, 0));
      for (int i : std::set<int>(gpus_.begin(), gpus_.end())) {
        if (cudaDeviceProps_[i].concurrentManagedAccess) {
          CUDA_RUNTIME(cudaMemAdvise(edgeSrc_d_, edgeBytes,
                                     cudaMemAdviseSetAccessedBy, i));
          CUDA_RUNTIME(cudaMemAdvise(edgeDst_d_, edgeBytes,
                                     cudaMemAdviseSetAccessedBy, i));
          CUDA_RUNTIME(cudaMemAdvise(nodes_d_, nodeBytes,
                                     cudaMemAdviseSetAccessedBy, i));
        } else {
          LOG(warn,
              "skipping cudaMemAdviseSetAccessedBy for device {}: "
              "cudaDeviceProp.concurrentManagedAccess = 0",
              i);
        }
      }
    }

    break;
  }
  case GPUMemoryKind::ZeroCopy: {
    // map host memory
    CUDA_RUNTIME(
        cudaHostRegister(hostDAG_.edgeSrc_.data(), edgeBytes,
                         cudaHostRegisterMapped | cudaHostRegisterPortable));
    CUDA_RUNTIME(
        cudaHostRegister(hostDAG_.edgeDst_.data(), edgeBytes,
                         cudaHostRegisterMapped | cudaHostRegisterPortable));
    CUDA_RUNTIME(
        cudaHostRegister(hostDAG_.nodes_.data(), nodeBytes,
                         cudaHostRegisterMapped | cudaHostRegisterPortable));

    // get valid device pointer
    CUDA_RUNTIME(
        cudaHostGetDevicePointer(&edgeSrc_d_, hostDAG_.edgeSrc_.data(), 0));
    CUDA_RUNTIME(
        cudaHostGetDevicePointer(&edgeDst_d_, hostDAG_.edgeDst_.data(), 0));
    CUDA_RUNTIME(
        cudaHostGetDevicePointer(&nodes_d_, hostDAG_.nodes_.data(), 0));

    // allocate memory for output
    CUDA_RUNTIME(
        cudaHostAlloc(&triangleCounts_, countBytes, cudaHostAllocMapped));
    break;
  }
  default: {
    LOG(critical, "unhandled value for gpu memory kind");
    exit(-1);
  }
  }

  nvtxRangePop();
}

size_t IMPACT2018TC::count() {
  nvtxRangePush(__PRETTY_FUNCTION__);
  const size_t numDev = gpus_.size();

  // split edges among devices
  size_t edgesPerDevice = (hostDAG_.num_edges() + numDev - 1) / numDev;
  SPDLOG_DEBUG(logger::console, "{} edges per GPU", edgesPerDevice);

  size_t edgeOffset = 0;
  for (int i : gpus_) {
    CUDA_RUNTIME(cudaSetDevice(i));

    size_t edgeCount =
        std::min(edgesPerDevice, hostDAG_.num_edges() - edgeOffset);
    SPDLOG_DEBUG(logger::console, "GPU {} edges {}+{}", i, edgeOffset,
                 edgeCount);

    dim3 dimBlock(256);
    size_t desiredGridSize = (edgeCount + dimBlock.x - 1) / dimBlock.x;
    dim3 dimGrid(
        std::min(size_t(std::numeric_limits<int>::max()), desiredGridSize));

    SPDLOG_DEBUG(logger::console, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
    kernel_tc<<<dimGrid, dimBlock>>>(triangleCounts_, edgeSrc_d_, edgeDst_d_,
                                     nodes_d_, edgeOffset, edgeCount);
    edgeOffset += edgesPerDevice;
  }

  for (int i : std::set<int>(gpus_.begin(), gpus_.end())) {
    CUDA_RUNTIME(cudaSetDevice(i));
    SPDLOG_DEBUG(logger::console, "Waiting for GPU {}", i);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  nvtxRangePush("final reduction");
  auto start = std::chrono::system_clock::now();
  size_t total = 0;
  for (size_t i = 0; i < hostDAG_.num_edges(); ++i) {
    total += triangleCounts_[i];
  }
  auto elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
  nvtxRangePop();
  SPDLOG_DEBUG(logger::console, "CPU reduction {}s", elapsed);

  nvtxRangePop();
  return total;
}

} // namespace pangolin