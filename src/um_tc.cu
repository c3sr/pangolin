#include "graph/um_tc.hpp"
#include "graph/logger.hpp"
#include "graph/utilities.hpp"
#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/dag2019.hpp"

#include <set>
#include <nvToolsExt.h>

__global__ static void kernel_tc(size_t *triangleCounts, const Int *edgeSrc, const Int *edgeDst, const Int *nodes, const size_t edgeOffset, const size_t numEdges){
     
    const Int gx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (Int i = gx + edgeOffset; i < edgeOffset + numEdges; i += blockDim.x * gridDim.x) {

        // get the src and dst node for this edge
        const Int src = edgeSrc[i];
        const Int dst = edgeDst[i];

        Int src_edge = nodes[src];
        const Int src_edge_end = nodes[src + 1];

        Int dst_edge = nodes[dst];
        const Int dst_edge_end = nodes[dst + 1];

        size_t count = 0;

        bool readSrc = true;
        bool readDst = true;

        while (src_edge < src_edge_end && dst_edge < dst_edge_end){

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

        triangleCounts[i] = count;
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


        dim3 dimBlock(512);
        dim3 dimGrid((edgeCount + dimBlock.x - 1) / dimBlock.x);
    
        LOG(debug, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
        kernel_tc<<<dimGrid, dimBlock>>>(triangleCounts_, edgeSrc_d_, edgeDst_d_, nodes_d_, edgeOffset, edgeCount);
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
