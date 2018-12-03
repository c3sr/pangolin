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


CSRTC::CSRTC(Config &c) {

    gpus_ = c.gpus_;

    for (auto i : gpus_) {
        LOG(info, "Initializing CUDA device {}", i);
        CUDA_RUNTIME(cudaSetDevice(i));
        CUDA_RUNTIME(cudaFree(0));
    }

    triangleCounts_ = std::vector<Uint *>(gpus_.size(), nullptr);
	rowStarts_d_.resize(gpus_.size());
	nonZeros_d_.resize(gpus_.size());
	isLocalNonZero_d_.resize(gpus_.size());
}

CSRTC::~CSRTC() {

    for (auto p : triangleCounts_) {
        CUDA_RUNTIME(cudaFreeHost(p));
    }
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
    // for singe dag, no remote edges
    auto graph = ParGraph::from_edges(filtered, EdgeList());

    LOG(info, "{} edges", graph.nnz());
    LOG(info, "{} nodes", graph.num_rows());

    if (gpus_.size() == 1) {
        graphs_.push_back(graph);
    } else {
        graphs_ = graph.partition_nonzeros(gpus_.size());
    }

    size_t partNodes = 0;
    size_t partEdges = 0;
    for (const auto &graph : graphs_) {
        partNodes += graph.num_rows();
        partEdges += graph.nnz();
    }
    LOG(info, "node replication {}", partNodes / graph.num_rows());
    LOG(info, "edge replication {}", partEdges / graph.nnz());
}

void CSRTC::setup_data() {
    assert(gpus_.size());
    assert(graphs_.size() == gpus_.size());
    for (size_t i = 0; i < graphs_.size(); ++i) {
        const auto &graph = graphs_[i];
        const int dev = gpus_[i];
        CUDA_RUNTIME(cudaSetDevice(dev));

        dimGrids_.push_back(graph.num_rows());
        const dim3 &dimGrid = dimGrids_[i];

        const size_t rowStartsSz = graph.rowStarts_.size() * sizeof(Int);
        const size_t nonZerosSz = graph.nonZeros_.size() * sizeof(Int);
        const size_t isLocalNonZeroSz = graph.isLocalNonZero_.size() * sizeof(graph.isLocalNonZero_[0]);
        const size_t countSz = dimGrid.x * sizeof(*(triangleCounts_[0]));

        auto &triangleCounts = triangleCounts_[i];
        auto &rowStarts_d = rowStarts_d_[i];
        auto &nonZeros_d = nonZeros_d_[i];
        auto &isLocalNonZero_d = isLocalNonZero_d_[i];


        CUDA_RUNTIME(cudaMalloc(&rowStarts_d, rowStartsSz));
        CUDA_RUNTIME(cudaMalloc(&nonZeros_d, nonZerosSz));
        CUDA_RUNTIME(cudaMalloc(&isLocalNonZero_d, isLocalNonZeroSz));
        CUDA_RUNTIME(cudaHostAlloc(&triangleCounts, countSz, cudaHostAllocMapped));

        LOG(trace, "copy {} rowStarts bytes", rowStartsSz);
        CUDA_RUNTIME(cudaMemcpy(rowStarts_d, graph.rowStarts_.data(), rowStartsSz, cudaMemcpyDefault));
        LOG(trace, "copy {} nonZeros bytes", nonZerosSz);
        CUDA_RUNTIME(cudaMemcpy(nonZeros_d, graph.nonZeros_.data(), nonZerosSz, cudaMemcpyDefault)); 
        LOG(trace, "copy {} isLocalNonZero bytes", isLocalNonZeroSz);
        CUDA_RUNTIME(cudaMemcpy(isLocalNonZero_d, graph.isLocalNonZero_.data(), isLocalNonZeroSz, cudaMemcpyDefault)); 
    }
}

size_t CSRTC::count() {

    assert(graphs_.size() == gpus_.size());
    for (size_t i = 0; i < graphs_.size(); ++i) {
        const auto &graph = graphs_[i];
        const int dev = gpus_[i];
        const dim3 &dimGrid = dimGrids_[i];
        
        const size_t numRows = graph.num_rows();
        dim3 dimBlock(BLOCK_DIM_X);
        LOG(debug, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
        kernel_tc2<<<dimGrid, dimBlock>>>(triangleCounts_[i], rowStarts_d_[i], nonZeros_d_[i], isLocalNonZero_d_[i], numRows);
    }


    Uint total = 0;
    for (size_t i = 0; i < graphs_.size(); ++i) {
        const int &dev = gpus_[i];
        const dim3 &dimGrid = dimGrids_[i];
        LOG(debug, "waiting for GPU {}", dev);
        CUDA_RUNTIME(cudaSetDevice(dev));
        CUDA_RUNTIME(cudaDeviceSynchronize());

        Uint partitionTotal = 0;
        LOG(debug, "cpu reduction for GPU {}", dev);
        for(size_t j = 0; j < dimGrid.x; ++j) {
            partitionTotal += triangleCounts_[i][j];
        }
        LOG(debug, "partition had {} triangles", partitionTotal);
        total += partitionTotal;
    }

    // auto start = std::chrono::system_clock::now();
    // for(size_t i = 0; i < dimGrid_.x; ++i) {
    //     total += triangleCounts_[i];
    // }
    // auto elapsed = (std::chrono::system_clock::now() - start).count() / 1e9;
    // LOG(debug, "CPU reduction {}s", elapsed);


    return total;
}
