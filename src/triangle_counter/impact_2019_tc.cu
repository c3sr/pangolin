/// \file

#include "pangolin/triangle_counter/impact_2019_tc.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"
#include "pangolin/reader/gc_tsv_reader.hpp"

#include <nvToolsExt.h>
#include <limits>
#include <cub/cub.cuh>

__device__ static size_t linear_intersection_count(const Int *const aBegin, const Int *const aEnd, const Int *const bBegin, const Int *const bEnd) {
    size_t count = 0;
    const Int *ap = aBegin;
    const Int *bp = bBegin;

    if (ap < aEnd && bp < bEnd) {

        bool loadA = false;
        bool loadB = false;
        Int a = *ap;
        Int b = *bp;
        
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
          }
          else if (a < b){
              ++ap;
              loadA = true;
          }
          else {
              ++bp;
              loadB = true;
          }
      }
    }
    return count;
}


/*! Count triangles

Use one thread per edge to count triangles.
Compare sorted neighbor lists linearly.
*/
__global__ static void kernel_tc(
    uint64_t * __restrict__ triangleCounts, //!< per-edge triangle counts
    const Int *const edgeSrc, //!< node ids for edge srcs
    const Int *const edgeDst, //!< node ids for edge dsts
    const Int *const nodes, //!< source node offsets in edgeDst
    const size_t edgeOffset, //!< where in the edge list this function should begin counting
    const size_t numEdges //!< how many edges to count triangles for
    ){
     
    const Int gx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (Int i = gx + edgeOffset; i < edgeOffset + numEdges; i += blockDim.x * gridDim.x) {

        // get the src and dst node for this edge
        const Int src = edgeSrc[i];
        const Int dst = edgeDst[i];

        const Int src_edge = nodes[src];
        const Int src_edge_end = nodes[src + 1];

        const Int dst_edge = nodes[dst];
        const Int dst_edge_end = nodes[dst + 1];

        size_t count = linear_intersection_count(&edgeDst[src_edge], &edgeDst[src_edge_end], &edgeDst[dst_edge], &edgeDst[dst_edge_end]);

        triangleCounts[i] = count;
    }
}


// return 1 if search_val is in array between offets left and right, inclusive
__device__ static bool binary_search(const Int *const array, size_t left,
    size_t right, const Int search_val) {
    while (left <= right) {
        size_t mid = (left + right) / 2;
        Int val = array[mid];
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

/*! Count triangles

Use one warp per edge to count triangles.
Compare neighbor lists in parallel with a binary search of the longer list

*/
template <size_t BLOCK_DIM_X>
__global__ static void
kernel_binary(
    uint64_t *__restrict__ edgeTriangleCounts, //<! per-edge triangle count
    const Int *edgeSrc,
    const Int *edgeDst,
    const Int *rowStarts, //<! offset in edgeSrc/edgeDst where each row starts
    const Int edgeOffset,
    const Int numEdges) {

  const size_t WARPS_PER_BLOCK = BLOCK_DIM_X / 32;
  static_assert(BLOCK_DIM_X % 32 ==
                0, "expect integer number of warps per block");
  typedef cub::WarpReduce<size_t> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];


  const int warpIdx = threadIdx.x / 32; // which warp in thread block
  const int laneIdx = threadIdx.x % 32; // which thread in warp


  const size_t gwIdx = warpIdx + blockIdx.x * WARPS_PER_BLOCK;

  // one warp per edge
  for (Int edgeIdx = gwIdx; edgeIdx < numEdges; edgeIdx += WARPS_PER_BLOCK * gridDim.x) {

    size_t count = 0;

    // head and tail of edge
    const Int head = edgeSrc[edgeIdx];
    const Int tail = edgeDst[edgeIdx];

    // neighbor offsets for head of edge
    const Int headOffStart = rowStarts[head];
    const Int headOffEnd = rowStarts[head + 1];

    // neighbor offsets for tail of edge
    const Int tailOffStart = rowStarts[tail];
    const Int tailOffEnd = rowStarts[tail + 1];

    if (headOffEnd - headOffStart < tailOffEnd - tailOffStart) {
        for (const Int *u = &edgeDst[headOffStart] + laneIdx;
            u < &edgeDst[headOffEnd]; u += 32) {
        count +=
            binary_search(edgeDst, tailOffStart, tailOffEnd - 1, *u);
        }
    } else {
        for (const Int *u = &edgeDst[tailOffStart] + laneIdx;
            u < &edgeDst[tailOffEnd]; u += 32) {
        count +=
            binary_search(edgeDst, headOffStart, headOffEnd - 1, *u);
        }
    }



    size_t aggregate = WarpReduce(temp_storage[warpIdx]).Sum(count);

    if (laneIdx == 0) {
        edgeTriangleCounts[edgeIdx] = aggregate;
    }

  }
}




IMPACT2019TC::IMPACT2019TC(Config &c)  : CUDATriangleCounter(c) {
    nvtxRangePush(__PRETTY_FUNCTION__);
    LOG(debug, "IMPACT 2019 TC, sizeof(Int) = {}", sizeof(Int));

    if (c.storage_ == "um") {
        GPUMemoryKind_ = GPUMemoryKind::Unified;
    } else if (c.storage_ == "zc") {
        GPUMemoryKind_ = GPUMemoryKind::ZeroCopy;
    } else {
        LOG(critical, "unknown gpu storage kind \"{}\"", c.storage_);
        exit(-1);
    }

    if ("linear" == c.kernel_) {
        kernelKind_ = KernelKind::Linear;
    } else if ("binary" == c.kernel_) {
        kernelKind_ = KernelKind::Binary;
    } else {
        LOG(critical, "unknown kernel kind \"{}\", expecting linear|binary", c.kernel_);
        exit(-1);
    }

    unifiedMemoryHints_ = c.hints_;
    nvtxRangePop();
}

IMPACT2019TC::~IMPACT2019TC() {
    nvtxRangePush(__PRETTY_FUNCTION__);
    switch (GPUMemoryKind_) {
        case GPUMemoryKind::Unified: {
            CUDA_RUNTIME(cudaFree(edgeSrc_d_));
            CUDA_RUNTIME(cudaFree(edgeDst_d_));
            CUDA_RUNTIME(cudaFree(cols_d_));
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

void IMPACT2019TC::read_data(const std::string &path) {
    nvtxRangePush(__PRETTY_FUNCTION__);
    LOG(info, "reading {}", path);
    auto *reader = pangolin::EdgeListReader::from_file(path);
    auto edgeList = reader->read();
    if (edgeList.size() == 0) {
        LOG(warn, "empty edge list");
    }
    LOG(debug, "building DAG");
    hostDAG_ = DAG2019::from_edgelist(edgeList);

    LOG(info, "{} nodes", hostDAG_.num_nodes());
    LOG(info, "{} edges", hostDAG_.num_edges());
    nvtxRangePop();
}

void IMPACT2019TC::setup_data() {
    nvtxRangePush(__PRETTY_FUNCTION__);
    const size_t edgeBytes = hostDAG_.edgeSrc_.size() * sizeof(Int);
    const size_t nodeBytes = hostDAG_.nodes_.size() * sizeof(Int);
    const size_t countBytes = hostDAG_.num_edges() * sizeof(*triangleCounts_);

    switch (GPUMemoryKind_) {
        case GPUMemoryKind::Unified: {
            CUDA_RUNTIME(cudaMallocManaged(&edgeSrc_d_, edgeBytes));
            CUDA_RUNTIME(cudaMallocManaged(&edgeDst_d_, edgeBytes));
            CUDA_RUNTIME(cudaMallocManaged(&cols_d_, nodeBytes));
            CUDA_RUNTIME(cudaMallocManaged(&triangleCounts_, countBytes));

            LOG(debug, "copying to unified memory");
            CUDA_RUNTIME(cudaMemcpy(edgeSrc_d_, hostDAG_.edgeSrc_.data(), edgeBytes, cudaMemcpyDefault));
            CUDA_RUNTIME(cudaMemcpy(edgeDst_d_, hostDAG_.edgeDst_.data(), edgeBytes, cudaMemcpyDefault)); 
            CUDA_RUNTIME(cudaMemcpy(cols_d_, hostDAG_.nodes_.data(), nodeBytes, cudaMemcpyDefault));

            // processor id is ignored
            if (unifiedMemoryHints_) {
                LOG(info, "using unified memory hints");
                CUDA_RUNTIME(cudaMemAdvise(edgeSrc_d_, edgeBytes, cudaMemAdviseSetReadMostly, 0));
                CUDA_RUNTIME(cudaMemAdvise(edgeDst_d_, edgeBytes, cudaMemAdviseSetReadMostly, 0));
                CUDA_RUNTIME(cudaMemAdvise(cols_d_, nodeBytes, cudaMemAdviseSetReadMostly, 0));
                for (int i : std::set<int>(gpus_.begin(), gpus_.end())) {
                    if (cudaDeviceProps_[i].concurrentManagedAccess) {
                        CUDA_RUNTIME(cudaMemAdvise(edgeSrc_d_, edgeBytes, cudaMemAdviseSetAccessedBy, i));
                        CUDA_RUNTIME(cudaMemAdvise(edgeDst_d_, edgeBytes, cudaMemAdviseSetAccessedBy, i));
                        CUDA_RUNTIME(cudaMemAdvise(cols_d_, nodeBytes, cudaMemAdviseSetAccessedBy, i));
                    } else {
                        LOG(warn, "skipping cudaMemAdviseSetAccessedBy for device {}: cudaDeviceProp.concurrentManagedAccess = 0", i);
                    }
                }
            }

            break;
        }
        case GPUMemoryKind::ZeroCopy: {
            // map host memory
            CUDA_RUNTIME(cudaHostRegister(hostDAG_.edgeSrc_.data(), edgeBytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
            CUDA_RUNTIME(cudaHostRegister(hostDAG_.edgeDst_.data(), edgeBytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
            CUDA_RUNTIME(cudaHostRegister(hostDAG_.nodes_.data(), nodeBytes, cudaHostRegisterMapped | cudaHostRegisterPortable));

            // get valid device pointer
            CUDA_RUNTIME(cudaHostGetDevicePointer(&edgeSrc_d_, hostDAG_.edgeSrc_.data(), 0));
            CUDA_RUNTIME(cudaHostGetDevicePointer(&edgeDst_d_, hostDAG_.edgeDst_.data(), 0)); 
            CUDA_RUNTIME(cudaHostGetDevicePointer(&cols_d_, hostDAG_.nodes_.data(), 0));

            // allocate memory for output
            CUDA_RUNTIME(cudaHostAlloc(&triangleCounts_, countBytes, cudaHostAllocMapped));
            break;
        }
        default: {
            LOG(critical, "unhandled value for gpu memory kind");
            exit(-1);
        }
    }

    nvtxRangePop();
}

size_t IMPACT2019TC::count() {
    nvtxRangePush(__PRETTY_FUNCTION__);
    const size_t numDev = gpus_.size();

    // split edges among devices
    size_t edgesPerDevice = (hostDAG_.num_edges() + numDev - 1) / numDev;
    LOG(debug, "{} edges per GPU", edgesPerDevice);

    size_t edgeOffset = 0;
    for (int i : gpus_) {
        CUDA_RUNTIME(cudaSetDevice(i));

        size_t edgeCount = std::min(edgesPerDevice, hostDAG_.num_edges() - edgeOffset);
        LOG(debug, "GPU {} edges {}+{}", i, edgeOffset, edgeCount);

        // Launch the correct kind of kernel
        switch (kernelKind_) {
            case KernelKind::Linear: {
                LOG(debug, "linear kernel");
                dim3 dimBlock(256);
                size_t desiredGridSize = (edgeCount + dimBlock.x - 1) / dimBlock.x;
                dim3 dimGrid(std::min(size_t(std::numeric_limits<int>::max()), desiredGridSize));
                LOG(debug, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
                kernel_tc<<<dimGrid, dimBlock>>>(triangleCounts_, edgeSrc_d_, edgeDst_d_, cols_d_, edgeOffset, edgeCount);
                break;
            }
            case KernelKind::Binary: {
                LOG(debug, "binary kernel");
                constexpr int dimBlock = 512;
                static_assert(dimBlock % 32 == 0, "Expect integer warps per block");
                const int warpsPerBlock = dimBlock / 32;
                size_t desiredGridSize = (edgeCount + warpsPerBlock - 1) / warpsPerBlock;
                dim3 dimGrid(std::min(size_t(std::numeric_limits<int>::max()), desiredGridSize));
                LOG(debug, "kernel dims {} x {}", dimGrid.x, dimBlock);
                kernel_binary<dimBlock><<<dimGrid, dimBlock>>>(triangleCounts_, edgeSrc_d_, edgeDst_d_, cols_d_, edgeOffset, edgeCount);
                break;
            }
            default: {
                LOG(critical, "unexpected kernelKind_");
                exit(-1);
            }
        }


        edgeOffset += edgesPerDevice;
    }
    
    for (int i : std::set<int>(gpus_.begin(), gpus_.end())) {
        CUDA_RUNTIME(cudaSetDevice(i));
        LOG(debug, "Waiting for GPU {}", i);
        CUDA_RUNTIME(cudaDeviceSynchronize());
    }

    nvtxRangePush("final reduction");
    auto start = std::chrono::system_clock::now();
    size_t total = 0;
    for(size_t i = 0; i < hostDAG_.num_edges(); ++i) {
        total += triangleCounts_[i];
    }
    auto elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    nvtxRangePop();
    LOG(debug, "CPU reduction {}s", elapsed);

    nvtxRangePop();
    return total;
}
