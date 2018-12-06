#include "graph/um_tc.hpp"
#include "graph/logger.hpp"
#include "graph/utilities.hpp"
#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/dag2019.hpp"

#include <set>
#include <nvToolsExt.h>
#include <cub/cub.cuh>

__device__ static bool binary_search(const Int* const array, Int left, Int right, const Int search_val) {
    while(left <= right) {
        int mid = (left + right)/2;
        int val = array[mid];
        if(val < search_val) {
            left = mid + 1;
        } else if(val > search_val) {
            right = mid - 1;
        } else { // val == search_val
            return 1;
        }
    }
    return 0;
}

__device__ static size_t intersection_count(const Int *const aBegin, const Int *const aEnd, const Int *const bBegin, const Int *const bEnd) {
    size_t count = 0;
    const Int *ap = aBegin;
    const Int *bp = bBegin;

    while (ap < aEnd && bp < bEnd) {

        if (*ap == *bp) {
            ++count;
            ++ap;
            ++bp;
        }
        else if (*ap < *bp){
            ++ap;
        }
        else {
            ++bp;
        }
    }
    return count;
}

template <size_t BLOCK_DIM_X>
__global__ static void kernel_tc(size_t * __restrict__ triangleCounts, const Int *edgeSrc, const Int *edgeDst, const Int *nodes, const size_t edgeOffset, const size_t numEdges){
     
    const Int gx = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    
    for (Int i = gx + edgeOffset; i < edgeOffset + numEdges; i += BLOCK_DIM_X * gridDim.x) {

        // get the src and dst node for this edge
        const Int src = edgeSrc[i];
        const Int dst = edgeDst[i];

        Int src_edge = nodes[src];
        const Int src_edge_end = nodes[src + 1];

        Int dst_edge = nodes[dst];
        const Int dst_edge_end = nodes[dst + 1];

        size_t count = 0;



            count = intersection_count(&edgeDst[src_edge], &edgeDst[src_edge_end], &edgeDst[dst_edge], &edgeDst[dst_edge_end]);



        /*
        bool readSrc = true;
        bool readDst = true;
        while (src_edge < src_edge_end && dst_edge < dst_edge_end) {

            Int u, v;

            if (readSrc) {
                u = edgeDst[src_edge];
                readSrc = false;
            }

            if (readDst) {
                v = edgeDst[dst_edge];
                readDst = false;
            }

            // the two nodes that make up this edge both have a common dst
            if (u == v) {
                ++count;
                ++src_edge;
                ++dst_edge;
                readSrc = true;
                readDst = true;
            }
            else if (u < v){
                ++src_edge;
                readSrc = true;
            }
            else {
                ++dst_edge;
                readDst = true;
            }
        }
        */

        triangleCounts[i] = count;
    }
}


template <size_t BLOCK_DIM_X>
__global__ static void kernel_tc2(size_t * __restrict__ triangleCounts, const Int *edgeSrc, const Int *edgeDst, const Int *nodes, const size_t edgeOffset, const size_t numEdges){

    static_assert(BLOCK_DIM_X > 0, "threadblock should have at least 1 thread");
    static_assert(BLOCK_DIM_X % 32 == 0, "require BLOCK_DIM_X to be an integer number of warps");
    const Int WARPS_PER_BLOCK = BLOCK_DIM_X / 32;

    const Int gx = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    const Int gwx = gx / 32;
    const Int lx = gx % 32;
    
    typedef cub::WarpReduce<size_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];

    // each edge gets a warp
    for (Int i = gwx + edgeOffset; i < edgeOffset + numEdges; i += WARPS_PER_BLOCK * gridDim.x) {

        // get the src and dst node for this edge
        #if 0
        Int src_edge, src_edge_end, dst_edge, dst_edge_end;
        if (lx == 0) {
            Int src = edgeSrc[i];
            Int dst = edgeDst[i];
            src_edge = nodes[src];
            src_edge_end = nodes[src + 1];
            dst_edge = nodes[dst];
            dst_edge_end = nodes[dst + 1];
        }
        src_edge = cub::ShuffleIndex<Int>(src_edge, 0, 32, 0xffffffff);
        dst_edge = cub::ShuffleIndex<Int>(dst_edge, 0, 32, 0xffffffff);
        src_edge_end = cub::ShuffleIndex<Int>(src_edge_end, 0, 32, 0xffffffff);
        dst_edge_end = cub::ShuffleIndex<Int>(dst_edge_end, 0, 32, 0xffffffff);
        #else

        // get the src and dst node for this edge
        const Int src = edgeSrc[i];
        const Int dst = edgeDst[i];
        const Int src_edge = nodes[src];
        const Int src_edge_end = nodes[src + 1];
        const Int dst_edge = nodes[dst];
        const Int dst_edge_end = nodes[dst + 1];
        #endif


        size_t count = 0;

        // binary search of larger list
        if (src_edge_end - src_edge < dst_edge_end - dst_edge) {
            for (const Int *u = &edgeDst[src_edge] + lx; u < &edgeDst[src_edge_end]; u += 32) {
                count += binary_search(edgeDst, dst_edge, dst_edge_end-1, *u);
            }
        } else {
            for (const Int *u = &edgeDst[dst_edge] + lx; u < &edgeDst[dst_edge_end]; u += 32) {
                count += binary_search(edgeDst, src_edge, src_edge_end-1, *u);
            }                
        }


        // Obtain one input item per thread
        // Return the warp-wide sums to each lane0 (threads 0, 32, 64, and 96)
        int warp_id = threadIdx.x / 32;
        size_t aggregate = WarpReduce(temp_storage[warp_id]).Sum(count);

        if (lx == 0) {
            triangleCounts[i] = aggregate;
        }
    }
}


UMTC::UMTC(Config &c) {
    nvtxRangePush(__PRETTY_FUNCTION__);
    LOG(debug, "ctor GPU triangle counter, sizeof(Int) = {}", sizeof(Int));

    gpus_ = c.gpus_;

    if (gpus_.empty()) {
        LOG(critical, "Unified-memory edge-set intersection triangle counter requires >= 1 GPU");
        exit(-1);
    }

    for (int dev : std::set<int>(gpus_.begin(), gpus_.end())) {
        LOG(info, "Initializing CUDA device {}", dev);
        CUDA_RUNTIME(cudaSetDevice(dev));
        CUDA_RUNTIME(cudaFree(0));
        if (0 == cudaDeviceProps_.count(dev)) {
            CUDA_RUNTIME(cudaGetDeviceProperties(&cudaDeviceProps_[dev], dev));
        }
    }
    nvtxRangePop();
}

UMTC::~UMTC() {
    nvtxRangePush(__PRETTY_FUNCTION__);
    CUDA_RUNTIME(cudaFree(edgeSrc_d_));
    CUDA_RUNTIME(cudaFree(edgeDst_d_));
    CUDA_RUNTIME(cudaFree(nodes_d_));
    CUDA_RUNTIME(cudaFree(triangleCounts_));
    nvtxRangePop();
}

void UMTC::read_data(const std::string &path) {
    nvtxRangePush(__PRETTY_FUNCTION__);
    LOG(info, "reading {}", path);
    GraphChallengeTSVReader reader(path);
    auto edgeList = reader.read_edges();
    LOG(debug, "building DAG");
    hostDAG_ = DAG2019::from_edgelist(edgeList);

    LOG(info, "{} nodes", hostDAG_.num_nodes());
    LOG(info, "{} edges", hostDAG_.num_edges());
    nvtxRangePop();
}

void UMTC::setup_data() {
    nvtxRangePush(__PRETTY_FUNCTION__);
    const size_t edgeBytes = hostDAG_.edgeSrc_.size() * sizeof(Int);
    const size_t nodeBytes = hostDAG_.nodes_.size() * sizeof(Int);
    const size_t countBytes = hostDAG_.num_edges() * sizeof(*triangleCounts_);

    LOG(debug, "allocating unified memory");
    CUDA_RUNTIME(cudaMallocManaged(&edgeSrc_d_, edgeBytes));
    CUDA_RUNTIME(cudaMallocManaged(&edgeDst_d_, edgeBytes));
    CUDA_RUNTIME(cudaMallocManaged(&nodes_d_, nodeBytes));
    CUDA_RUNTIME(cudaMallocManaged(&triangleCounts_, countBytes));

    LOG(debug, "copying to unified memory");
    CUDA_RUNTIME(cudaMemcpy(edgeSrc_d_, hostDAG_.edgeSrc_.data(), edgeBytes, cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpy(edgeDst_d_, hostDAG_.edgeDst_.data(), edgeBytes, cudaMemcpyDefault)); 
    CUDA_RUNTIME(cudaMemcpy(nodes_d_, hostDAG_.nodes_.data(), nodeBytes, cudaMemcpyDefault));

    // processor id is ignored
    CUDA_RUNTIME(cudaMemAdvise(edgeSrc_d_, edgeBytes, cudaMemAdviseSetReadMostly, 0));
    CUDA_RUNTIME(cudaMemAdvise(edgeDst_d_, edgeBytes, cudaMemAdviseSetReadMostly, 0));
    CUDA_RUNTIME(cudaMemAdvise(nodes_d_, nodeBytes, cudaMemAdviseSetReadMostly, 0));
    for (int i : std::set<int>(gpus_.begin(), gpus_.end())) {
      if (cudaDeviceProps_[i].concurrentManagedAccess) {
        CUDA_RUNTIME(cudaMemAdvise(edgeSrc_d_, edgeBytes, cudaMemAdviseSetAccessedBy, i));
        CUDA_RUNTIME(cudaMemAdvise(edgeDst_d_, edgeBytes, cudaMemAdviseSetAccessedBy, i));
        CUDA_RUNTIME(cudaMemAdvise(nodes_d_, nodeBytes, cudaMemAdviseSetAccessedBy, i));
      } else {
          LOG(warn, "skipping cudaMemAdviseSetAccessedBy for device {}: cudaDeviceProp.concurrentManagedAccess = 0", i);
      }
    }
    nvtxRangePop();
}

size_t UMTC::count() {
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

        const size_t BLOCK_SIZE = 128;
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid((edgeCount + dimBlock.x - 1) / dimBlock.x);
    
        LOG(debug, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
        kernel_tc2<BLOCK_SIZE><<<dimGrid, dimBlock>>>(triangleCounts_, edgeSrc_d_, edgeDst_d_, nodes_d_, edgeOffset, edgeCount);
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
