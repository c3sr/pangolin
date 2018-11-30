 #include "graph/zc_tc.hpp"
 #include "graph/logger.hpp"
 #include "graph/utilities.hpp"

 #include "graph/dag2019.hpp"

__global__ static void kernel_tc(Uint * __restrict__ triangleCounts, const Int *edgeSrc, const Int *edgeDst, const Int *nodes, const size_t edgeOffset, const size_t numEdges){
     
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

        bool uRead = true;
        bool vRead = true;
        Int u, v;

        while (src_edge < src_edge_end && dst_edge < dst_edge_end){

            if (uRead) {
                u = edgeDst[src_edge];
            }
            if (vRead) {
                v = edgeDst[dst_edge];
            }


            // the two nodes that make up this edge both have a common dst
            if (u == v) {
                ++count;
                ++src_edge;
                uRead = true;
                ++dst_edge;
                vRead = true;
            }
            else if (u < v){
                ++src_edge;
                uRead = true;
            }
            else {
                ++dst_edge;
                vRead = true;
            }
        }

        triangleCounts[i] = count;
    }
}

ZeroCopyTriangleCounter::ZeroCopyTriangleCounter() {
    LOG(debug, "ctor ZeroCopy triangle counter, sizeof(Int) = {}", sizeof(Int));

    int numDev;
    CUDA_RUNTIME(cudaGetDeviceCount(&numDev));
    for (int i = 0; i < numDev; ++i) {
        LOG(info, "Initializing CUDA device {}", i);
        CUDA_RUNTIME(cudaSetDevice(i));
        CUDA_RUNTIME(cudaFree(0));
    }


    }

ZeroCopyTriangleCounter::~ZeroCopyTriangleCounter() {
    LOG(debug, "dtor ZeroCopy triangle counter");
    LOG(debug, "unregistering/freeing CUDA memory");
    CUDA_RUNTIME(cudaHostUnregister(hostDAG_.edgeSrc_.data()));
    CUDA_RUNTIME(cudaHostUnregister(hostDAG_.edgeDst_.data()));
    CUDA_RUNTIME(cudaHostUnregister(hostDAG_.nodes_.data()));
    CUDA_RUNTIME(cudaFreeHost(triangleCounts_));
}

void ZeroCopyTriangleCounter::read_data(const std::string &path) {

    LOG(info, "reading {}", path);
    auto edgeList = EdgeList::read_tsv(path);
    LOG(debug, "building DAG");
    hostDAG_ = DAG2019::from_edgelist(edgeList);

    LOG(info, "{} nodes", hostDAG_.num_nodes());
    LOG(info, "{} edges", hostDAG_.num_edges());
}

void ZeroCopyTriangleCounter::setup_data() {
    const size_t edgeBytes = hostDAG_.edgeSrc_.size() * sizeof(Int);
    const size_t nodeBytes = hostDAG_.nodes_.size() * sizeof(Int);
    const size_t countBytes = hostDAG_.num_edges() * sizeof(*triangleCounts_);

    LOG(debug, "registering {}B", edgeBytes);
    CUDA_RUNTIME(cudaHostRegister(hostDAG_.edgeSrc_.data(), edgeBytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
    LOG(debug, "registering {}B", edgeBytes);
    CUDA_RUNTIME(cudaHostRegister(hostDAG_.edgeDst_.data(), edgeBytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
    LOG(debug, "registering {}B", nodeBytes);
    CUDA_RUNTIME(cudaHostRegister(hostDAG_.nodes_.data(), nodeBytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
    LOG(debug, "alloc/mapping {}B", countBytes);
    CUDA_RUNTIME(cudaHostAlloc(&triangleCounts_, countBytes, cudaHostAllocMapped));

    CUDA_RUNTIME(cudaHostGetDevicePointer(&edgeSrc_d_, hostDAG_.edgeSrc_.data(), 0));
    CUDA_RUNTIME(cudaHostGetDevicePointer(&edgeDst_d_, hostDAG_.edgeDst_.data(), 0)); 
    CUDA_RUNTIME(cudaHostGetDevicePointer(&nodes_d_, hostDAG_.nodes_.data(), 0));

}

Uint ZeroCopyTriangleCounter::count() {
    
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


        dim3 dimBlock(512);
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
    Uint total = 0;
    for(size_t i = 0; i < hostDAG_.num_edges(); ++i) {
        total += triangleCounts_[i];
    }
    auto elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(debug, "CPU reduction {}s", elapsed);


    return total;
}
