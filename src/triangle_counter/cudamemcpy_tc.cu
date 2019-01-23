#include "pangolin/triangle_counter/cudamemcpy_tc.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"
#include "pangolin/reader/gc_tsv_reader.hpp"

const int BLOCK_DIM_X = 128;

__global__ static void kernel_tc(size_t * __restrict__ triangleCounts, 
    const Int *edgeSrc, 
    const Int *edgeDst, 
    const Int *nodes,
    const bool *localDst, // if an edge is local ( should be counted)
    const size_t edgeOffset, 
    const size_t numEdges){
     

/*

Some edges may not be local.
These edges are needed to correctly count local triangles, but should not be counted as the base of a triangle themselves

*/

    const Int gx = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    
    for (Int i = gx + edgeOffset; i < edgeOffset + numEdges; i += BLOCK_DIM_X * gridDim.x) {

        // get the src and dst node for this edge
        const Int src = edgeSrc[i];
        const Int dst = edgeDst[i];
        size_t count = 0;
        if (localDst && localDst[i]) {
            Int src_edge = nodes[src];
            const Int src_edge_end = nodes[src + 1];
    
            Int dst_edge = nodes[dst];
            const Int dst_edge_end = nodes[dst + 1];
    
    
            while (src_edge < src_edge_end && dst_edge < dst_edge_end){
    
                Int u = edgeDst[src_edge];
                Int v = edgeDst[dst_edge];
    
                // the two nodes that make up this edge both have a common dst
                if (u == v) {
                    ++count;
                    ++src_edge;
                    ++dst_edge;
                }
                else if (u < v){
                    ++src_edge;
                }
                else {
                    ++dst_edge;
                }
            }
        } 

        triangleCounts[i] = count;
    }
}

CudaMemcpyTC::CudaMemcpyTC() {
    LOG(debug, "ctor GPU triangle counter, sizeof(Int) = {}", sizeof(Int));

    int numDev;
    CUDA_RUNTIME(cudaGetDeviceCount(&numDev));
    for (int i = 0; i < numDev; ++i) {
        LOG(info, "Initializing CUDA device {}", i);
        CUDA_RUNTIME(cudaSetDevice(i));
        CUDA_RUNTIME(cudaFree(0));
    }


    }

CudaMemcpyTC::~CudaMemcpyTC() {
    LOG(debug, "dtor GPU triangle counter");
    LOG(debug, "unregistering/freeing CUDA memory");
    CUDA_RUNTIME(cudaFree(edgeSrc_d_));
    CUDA_RUNTIME(cudaFree(edgeDst_d_));
    CUDA_RUNTIME(cudaFree(nodes_d_));
    CUDA_RUNTIME(cudaFreeHost(triangleCounts_));
}

void CudaMemcpyTC::read_data(const std::string &path) {

    LOG(info, "reading {}", path);
    auto *reader = pangolin::EdgeListReader::from_file(path);
    auto edgeList = reader->read();
    LOG(debug, "building DAG");
    hostDAG_ = DAG2019::from_edgelist(edgeList);

    LOG(info, "{} nodes", hostDAG_.num_nodes());
    LOG(info, "{} edges", hostDAG_.num_edges());
}

void CudaMemcpyTC::setup_data() {
    const size_t edgeBytes = hostDAG_.edgeSrc_.size() * sizeof(Int);
    const size_t nodeBytes = hostDAG_.nodes_.size() * sizeof(Int);
    const size_t countBytes = hostDAG_.num_edges() * sizeof(*triangleCounts_);

    CUDA_RUNTIME(cudaMalloc(&edgeSrc_d_, edgeBytes));
    CUDA_RUNTIME(cudaMalloc(&edgeDst_d_, edgeBytes));
    CUDA_RUNTIME(cudaMalloc(&nodes_d_, nodeBytes));
    CUDA_RUNTIME(cudaHostAlloc(&triangleCounts_, countBytes, cudaHostAllocMapped));

    CUDA_RUNTIME(cudaMemcpy(edgeSrc_d_, hostDAG_.edgeSrc_.data(), edgeBytes, cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpy(edgeDst_d_, hostDAG_.edgeDst_.data(), edgeBytes, cudaMemcpyDefault)); 
    CUDA_RUNTIME(cudaMemcpy(nodes_d_, hostDAG_.nodes_.data(), nodeBytes, cudaMemcpyDefault));

}

size_t CudaMemcpyTC::count() {
    
    int numDev;
    CUDA_RUNTIME(cudaGetDeviceCount(&numDev));

    // split edges into devices
    size_t edgesPerDevice = (hostDAG_.num_edges() + numDev - 1) / numDev;
    LOG(debug, "{} edges per GPU", edgesPerDevice);

    size_t edgeOffset = 0;
    for (int i = 0; i < numDev; ++i) {
        CUDA_RUNTIME(cudaSetDevice(i));


        size_t edgeCount = std::min(edgesPerDevice, hostDAG_.num_edges() - edgeOffset);
        LOG(debug, "GPU {} edges {}+{}", i, edgeOffset, edgeCount);


        dim3 dimBlock(BLOCK_DIM_X);
        dim3 dimGrid((hostDAG_.num_edges() + dimBlock.x - 1) / dimBlock.x);
    
        LOG(debug, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
        kernel_tc<<<dimGrid, dimBlock>>>(triangleCounts_, edgeSrc_d_, edgeDst_d_, nodes_d_, nullptr, edgeOffset, edgeCount);
        edgeOffset += edgesPerDevice;
    }
    
    for (int i = 0; i < numDev; ++i) {
        CUDA_RUNTIME(cudaSetDevice(i));
        LOG(debug, "Waiting for GPU {}", i);
        CUDA_RUNTIME(cudaDeviceSynchronize());
    }

    auto start = std::chrono::system_clock::now();
    size_t total = 0;
    for(size_t i = 0; i < hostDAG_.num_edges(); ++i) {
        total += triangleCounts_[i];
    }
    auto elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(debug, "CPU reduction {}s", elapsed);


    return total;
}
