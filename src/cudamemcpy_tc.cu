 #include "graph/cudamemcpy_tc.hpp"
 #include "graph/logger.hpp"
 #include "graph/utilities.hpp"

 #include "graph/dag2019.hpp"

const int BLOCK_DIM_X = 128;

__global__ static void kernel_tc(size_t * __restrict__ triangleCounts, const Int *edgeSrc, const Int *edgeDst, const Int *nodes, const size_t edgeOffset, const size_t numEdges){
     
    const Int gx = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    
    for (Int i = gx + edgeOffset; i < edgeOffset + numEdges; i += BLOCK_DIM_X * gridDim.x) {

        const Int src = edgeSrc[i];
        const Int dst = edgeDst[i];

        Int src_edge = nodes[src];
        const Int src_edge_end = nodes[src + 1];

        Int dst_edge = nodes[dst];
        const Int dst_edge_end = nodes[dst + 1];


        size_t count = 0;

        bool update_u = true;
        bool update_v = true;
        while (src_edge < src_edge_end && dst_edge < dst_edge_end){
            Int u, v;
            if (update_u) u = edgeDst[src_edge];
            if (update_v) v = edgeDst[dst_edge];

            // the two nodes that make up this edge both have a common dst
            if (u == v) {
                ++count;
                ++src_edge;
                ++dst_edge;
                update_u = true;
                update_v = true;
            }
            else if (u < v){
                ++src_edge;
                update_u = true;
                update_v = false;
            }
            else {
                ++dst_edge;
                update_u = false;
                update_v = true;
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
    auto edgeList = EdgeList::read_tsv(path);
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
        kernel_tc<<<dimGrid, dimBlock>>>(triangleCounts_, edgeSrc_d_, edgeDst_d_, nodes_d_, edgeOffset, edgeCount);
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
