#include "graph/vertex_tc.hpp"
#include "graph/logger.hpp"
#include "graph/utilities.hpp"
#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/par_graph.hpp"

#include <cub/cub.cuh>

const int BLOCK_DIM_X = 128;

__global__ static void kernel_tc2(
    uint64_t * __restrict__ triangleCounts, // per block triangle count
    const Uint *rowStarts,
    const Uint *nonZeros,
    const char *isLocalNonZero,
    const size_t numRows) {
     

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
    size_t aggregate = BlockReduce(temp_storage).Sum(count);
    if (threadIdx.x == 0) {
        triangleCounts[blockIdx.x] = aggregate;
    }
    
}


VertexTC::VertexTC(Config &c) : CUDATriangleCounter(c) {
    triangleCounts_d_ = std::vector<uint64_t*>(gpus_.size(), nullptr);
	rowOffsets_d_.resize(gpus_.size());
	nonZeros_d_.resize(gpus_.size());
	isLocalNonZero_d_.resize(gpus_.size());
}

VertexTC::~VertexTC() {
}

void VertexTC::read_data(const std::string &path) {

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

    // subtract 1 from edges
    // for (auto &e : filtered) {
    //     e.first -= 1;
    //     e.second -= 1;
    // }

    LOG(trace, "filtered edge list has {} entries", filtered.size());

    LOG(debug, "building DAG");
    // for singe dag, no remote edges
    auto graph = UnifiedMemoryCSR::from_sorted_edgelist(filtered);

    numEdges_ = graph.nnz();
    numNodes_ = graph.num_rows();
    LOG(info, "{} edges", numEdges_);
    LOG(info, "{} nodes", numNodes_);

    if (gpus_.size() == 1) {
        graphs_.push_back(graph);
    } else {
        graphs_ = graph.partition_nonzeros(gpus_.size());
    }

    if (graphs_.size() > 1) {
        size_t partNodes = 0;
        size_t partEdges = 0;
        for (const auto &graph : graphs_) {
            partNodes += graph.num_rows();
            partEdges += graph.nnz();
        }
        LOG(info, "node replication {}", partNodes / graph.num_rows());
        LOG(info, "edge replication {}", partEdges / graph.nnz());
    }
}

void VertexTC::setup_data() {
    assert(gpus_.size());
    assert(graphs_.size() == gpus_.size());

    // one set of triangle counts per partition
    triangleCounts_.resize(gpus_.size());

    for (size_t i = 0; i < graphs_.size(); ++i) {
        const auto &graph = graphs_[i];
        const int dev = gpus_[i];
        CUDA_RUNTIME(cudaSetDevice(dev));

        dimGrids_.push_back(graph.num_rows());
        const dim3 &dimGrid = dimGrids_[i];

        // create space for one triangle count per block
        
        triangleCounts_[i].resize(dimGrid.x, 0);

        // device pointers are directly the managed memory pointers
        triangleCounts_d_[i] = triangleCounts_[i].data();
        rowOffsets_d_[i] = graph.row_offsets();
        nonZeros_d_[i] = graph.cols();
        isLocalNonZero_d_[i] = graph.is_local_cols();
    }

    for (const auto i : gpus_) {
        LOG(trace, "synchronizing GPU {}", i);
        CUDA_RUNTIME(cudaSetDevice(i));
        CUDA_RUNTIME(cudaDeviceSynchronize());
    }
    LOG(trace, "here");
}

size_t VertexTC::count() {
    assert(graphs_.size() == gpus_.size());
    for (size_t i = 0; i < graphs_.size(); ++i) {
        const auto &graph = graphs_[i];
        const int dev = gpus_[i];
        const dim3 &dimGrid = dimGrids_[i];
        
        const size_t numRows = graph.num_rows();
        dim3 dimBlock(BLOCK_DIM_X);
        LOG(debug, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
        kernel_tc2<<<dimGrid, dimBlock>>>(triangleCounts_d_[i], rowOffsets_d_[i], nonZeros_d_[i], isLocalNonZero_d_[i], numRows);
        CUDA_RUNTIME(cudaGetLastError());
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
