#include "graph/vertex_tc.hpp"
#include "graph/logger.hpp"
#include "graph/utilities.hpp"
#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/par_graph.hpp"

#include <cub/cub.cuh>


// count intersections between sorted lists a and b
__device__ static size_t linear_intersection_count(const Uint *const aBegin, const Uint *const aEnd, const Uint *const bBegin, const Uint *const bEnd) {
    size_t count = 0;
    const auto *ap = aBegin;
    const auto *bp = bBegin;

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



template<size_t BLOCK_DIM_X>
__global__ static void kernel_linear(
    uint64_t * __restrict__ triangleCounts, // per block triangle count
    const Uint *rowStarts,
    const Uint *nonZeros,
    const char *isLocalNonZero,
    const size_t numRows
) {
     
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
                

                // edges from the head 
                const Uint headEdgeStart = tailStart;
                const Uint headEdgeEnd = tailEnd;
        
                // edge from the tail
                const Uint tail = nonZeros[tailOff];
                const Uint tailEdgeStart = rowStarts[tail];
                const Uint tailEdgeEnd = rowStarts[tail + 1];


                count += linear_intersection_count(&nonZeros[headEdgeStart], &nonZeros[headEdgeEnd], &nonZeros[tailEdgeStart], &nonZeros[tailEdgeEnd]);
            } 
        }          
    }

    size_t aggregate = BlockReduce(temp_storage).Sum(count);
    if (threadIdx.x == 0) {
        triangleCounts[blockIdx.x] = aggregate;
    }
    
}


template<size_t BLOCK_DIM_X>
__global__ static void kernel_linear_shared(
    uint64_t * __restrict__ triangleCounts, // per block triangle count
    const Uint *rowStarts,
    const Uint *nonZeros,
    const char *isLocalNonZero,
    const size_t numRows
) {
     
    // Specialize BlockReduce for a 1D block
    typedef cub::BlockReduce<size_t, BLOCK_DIM_X> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ Uint headDsts[1024];
   
    size_t count = 0;
    // one row per block
    // all threads in the block share the same output edges from the head
    for (Uint row = blockIdx.x; row < numRows; row += gridDim.x) {

        // offsets for head of edge
        const Uint head = row;

        // offsets for tail of edge
        const Uint tailStart = rowStarts[head];
        const Uint tailEnd = rowStarts[head+1];

        // pointers to beginning and end of row's nonzero columns
        const Uint *headEdgeBegin, *headEdgeEnd;

        // collaboratively load edges from head if they fit in shared memory
        if (tailEnd - tailStart < 1024) {
            for (size_t i = threadIdx.x + tailStart; i < tailEnd; i += BLOCK_DIM_X) {
                headDsts[i - tailStart] = nonZeros[i];
            }
            __syncthreads();
            headEdgeBegin = &headDsts[0];
            headEdgeEnd = &headDsts[tailEnd - tailStart];
        } else {
            headEdgeBegin = &nonZeros[tailStart];
            headEdgeEnd = &nonZeros[tailEnd];
        }
        

        // one thread per edge
        for (Uint tailOff = tailStart + threadIdx.x; tailOff < tailEnd; tailOff += BLOCK_DIM_X) {

            // only count local edges
            if (!isLocalNonZero || isLocalNonZero[tailOff]) {
                       
                // edge from the tail
                const Uint tail = nonZeros[tailOff];
                const Uint tailEdgeStart = rowStarts[tail];
                const Uint tailEdgeEnd = rowStarts[tail + 1];


                count += linear_intersection_count(headEdgeBegin, headEdgeEnd, &nonZeros[tailEdgeStart], &nonZeros[tailEdgeEnd]);
            } 
        }          
    }

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
        const size_t BLOCK_DIM_X = 128;
        dim3 dimBlock(BLOCK_DIM_X);
        LOG(debug, "kernel dims {} x {}", dimGrid.x, dimBlock.x);
        kernel_linear_shared<BLOCK_DIM_X><<<dimGrid, dimBlock>>>(triangleCounts_d_[i], rowOffsets_d_[i], nonZeros_d_[i], isLocalNonZero_d_[i], numRows);
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
