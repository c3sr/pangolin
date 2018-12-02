 #include "graph/csr_tc.hpp"
 #include "graph/logger.hpp"
 #include "graph/utilities.hpp"

 #include "graph/par_graph.hpp"

 #include <cub/cub.cuh>

const int BLOCK_DIM_X = 128;

__global__ static void kernel_tc(
    Uint * __restrict__ triangleCounts, // per block triangle count
    const Int *rowStarts,
    const Int *nonZeros,
    const bool *isLocalNonZero,
    const size_t numRows) {
     

    // Specialize BlockReduce for a 1D block of 128 threads on type int
    typedef cub::BlockReduce<Uint, BLOCK_DIM_X> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Compute the block-wide sum for thread0



    const Int gx = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    
    Uint count = 0;
    for (Int row = gx; row < numRows; row += BLOCK_DIM_X * gridDim.x) {

        // offsets for head of edge
        const Int head = row;

        // offsets for tail of edge
        const Int tailStart = rowStarts[head];
        const Int tailEnd = rowStarts[head+1];

        for (Int tailOff = tailStart; tailOff < tailEnd; ++tailOff) {

            // only count local edges
            if (!isLocalNonZero || isLocalNonZero[tailOff]) {
                const Int tail = nonZeros[tailOff];

                // edges from the head
                Int headEdge = tailStart;
                const Int headEdgeEnd = tailEnd;
        
                // edge from the tail
                Int tailEdge = rowStarts[tail];
                const Int tailEdgeEnd = rowStarts[tail + 1];
        
                bool readHead = true;
                bool readTail = true;
                while (headEdge < headEdgeEnd && tailEdge < tailEdgeEnd){
        
                    Int u, v;
                    if (readHead) {
                        u = nonZeros[headEdge];
                        readHead = false;
                    }
                    if (readTail) {
                        v = nonZeros[tailEdge];
                        readTail = false;
                    }

        
                    // the two nodes that make up this edge both have a common dst
                    if (u == v) {
                        ++count;
                        ++headEdge;
                        ++tailEdge;
                        readHead = true;
                        readTail = true;
                    }
                    else if (u < v){
                        ++headEdge;
                        readHead = true;
                    }
                    else {
                        ++tailEdge;
                        readTail = true;
                    }
                }
            } 
        }          
    }

    // if (threadIdx.x == 0) {
    //     triangleCounts[blockIdx.x] = 0;
    // }
    // __syncthreads();
    // atomicAdd(&triangleCounts[blockIdx.x], count);
    Uint aggregate = BlockReduce(temp_storage).Sum(count);
    if (threadIdx.x == 0) {
        triangleCounts[blockIdx.x] = aggregate;
    }
    
}

__global__ static void kernel_tc2(
    Uint * __restrict__ triangleCounts, // per block triangle count
    const Int *rowStarts,
    const Int *nonZeros,
    const bool *isLocalNonZero,
    const size_t numRows) {
     

    // Specialize BlockReduce for a 1D block of 128 threads on type int
    typedef cub::BlockReduce<Uint, BLOCK_DIM_X> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
   
    Uint count = 0;
    // one row per block
    for (Int row = blockIdx.x; row < numRows; row += gridDim.x) {

        // offsets for head of edge
        const Int head = row;

        // offsets for tail of edge
        const Int tailStart = rowStarts[head];
        const Int tailEnd = rowStarts[head+1];

        // one thread per edge
        for (Int tailOff = tailStart + threadIdx.x; tailOff < tailEnd; tailOff += BLOCK_DIM_X) {

            // only count local edges
            if (!isLocalNonZero || isLocalNonZero[tailOff]) {
                const Int tail = nonZeros[tailOff];

                // edges from the head
                Int headEdge = tailStart;
                const Int headEdgeEnd = tailEnd;
        
                // edge from the tail
                Int tailEdge = rowStarts[tail];
                // printf("tid=%d edge %d-%d reading nodes %d %d\n", threadIdx.x, head, tail, tail, tail+1);
                const Int tailEdgeEnd = rowStarts[tail + 1];
        
                bool readHead = true;
                bool readTail = true;
                while (headEdge < headEdgeEnd && tailEdge < tailEdgeEnd){
        
                    Int u, v;
                    if (readHead) {
                        u = nonZeros[headEdge];
                        readHead = false;
                    }
                    if (readTail) {
                        v = nonZeros[tailEdge];
                        readTail = false;
                    }

        
                    // the two nodes that make up this edge both have a common dst
                    if (u == v) {
                        ++count;
                        ++headEdge;
                        ++tailEdge;
                        readHead = true;
                        readTail = true;
                    }
                    else if (u < v){
                        ++headEdge;
                        readHead = true;
                    }
                    else {
                        ++tailEdge;
                        readTail = true;
                    }
                }
            } 
        }          
    }

    // if (threadIdx.x == 0) {
    //     triangleCounts[blockIdx.x] = 0;
    // }
    // __syncthreads();
    // atomicAdd(&triangleCounts[blockIdx.x], count);
    Uint aggregate = BlockReduce(temp_storage).Sum(count);
    if (threadIdx.x == 0) {
        triangleCounts[blockIdx.x] = aggregate;
    }
    
}


CSRTC::CSRTC(Config &c) : dimGrid_(1000) {

    gpus_ = c.gpus_;

    for (auto i : gpus_) {
        LOG(info, "Initializing CUDA device {}", i);
        CUDA_RUNTIME(cudaSetDevice(i));
        CUDA_RUNTIME(cudaFree(0));
    }
}

CSRTC::~CSRTC() {
    CUDA_RUNTIME(cudaFree(rowStarts_d_));
    CUDA_RUNTIME(cudaFree(nonZeros_d_));
    CUDA_RUNTIME(cudaFree(isLocalNonZero_d_));
    CUDA_RUNTIME(cudaFreeHost(triangleCounts_));
}

void CSRTC::read_data(const std::string &path) {

    LOG(info, "reading {}", path);
    auto edgeList = EdgeList::read_tsv(path);

    // convert edge list to DAG by src < dst
    EdgeList filtered;
    for (const auto &e : edgeList) {
        if (e.src_ < e.dst_) {
            filtered.push_back(e);
        }
    }

    LOG(debug, "building DAG");
    // remote is empty for now
    graph_ = ParGraph::from_edges(filtered, EdgeList());

    LOG(info, "{} edges", graph_.nnz());
    LOG(info, "{} nodes", graph_.num_nodes());
    LOG(info, "{} rows", graph_.rowStarts_.size() - 1);
}

void CSRTC::setup_data() {
    assert(gpus_.size());
    CUDA_RUNTIME(cudaSetDevice(gpus_[0]));
    const size_t rowStartsSz = graph_.rowStarts_.size() * sizeof(Int);
    const size_t nonZerosSz = graph_.nonZeros_.size() * sizeof(Int);
    const size_t isLocalNonZeroSz = 0;
    const size_t countSz = dimGrid_.x * sizeof(*triangleCounts_);

    CUDA_RUNTIME(cudaMalloc(&rowStarts_d_, rowStartsSz));
    CUDA_RUNTIME(cudaMalloc(&nonZeros_d_, nonZerosSz));
    isLocalNonZero_d_ = nullptr;
    CUDA_RUNTIME(cudaHostAlloc(&triangleCounts_, countSz, cudaHostAllocMapped));

    LOG(trace, "copy {} rowStarts_ bytes", rowStartsSz);
    CUDA_RUNTIME(cudaMemcpy(rowStarts_d_, graph_.rowStarts_.data(), rowStartsSz, cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpy(nonZeros_d_, graph_.nonZeros_.data(), nonZerosSz, cudaMemcpyDefault)); 

}

size_t CSRTC::count() {
    

        CUDA_RUNTIME(cudaSetDevice(0));
        const size_t numRows = graph_.num_rows();


        // size_t edgeCount = std::min(edgesPerDevice, graph_.num_edges() - edgeOffset);
        // LOG(debug, "GPU {} edges {}+{}", i, edgeOffset, edgeCount);


        dim3 dimBlock(BLOCK_DIM_X);
    
        LOG(debug, "kernel dims {} x {}", dimGrid_.x, dimBlock.x);
        kernel_tc2<<<dimGrid_, dimBlock>>>(triangleCounts_, rowStarts_d_, nonZeros_d_, isLocalNonZero_d_, numRows);

        CUDA_RUNTIME(cudaDeviceSynchronize());

    auto start = std::chrono::system_clock::now();
    size_t total = 0;
    for(size_t i = 0; i < dimGrid_.x; ++i) {
        total += triangleCounts_[i];
    }
    auto elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    LOG(debug, "CPU reduction {}s", elapsed);


    return total;
}
