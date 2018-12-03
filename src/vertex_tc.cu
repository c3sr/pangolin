#include "graph/vertex_tc.hpp"
#include "graph/logger.hpp"
#include "graph/utilities.hpp"
#include "graph/dag_lowertriangular_csr.hpp"
#include "graph/reader/gc_tsv_reader.hpp"


__device__ size_t intersections(const Int *a_b, const Int *a_e, const Int *b_b, const Int *b_e) {
    size_t count = 0;
    for (const Int *a = a_b; a != a_e; ++a) {
        if (*a == -1) {
            break;
        }
        for (const Int *b = b_b; b != b_e; ++b) {
            if (*b == -1) {
                break;
            }
            if (*a == *b) {
                // printf("%d\n", *a);
                ++count;
            }
        }
    }
    return count;
}

// cols: nonzeros in each row
// roff: starting offset of each row
const size_t BLOCK_DIM = 128;
__global__ static void kernel_tc(int * __restrict__ blockTriangleCounts, const Int *cols, const Int *roff, const size_t numRows){
     
    int count = 0;

    // one block per vertex (u)
    for (size_t u = blockIdx.x; u < numRows; u += gridDim.x) {
        
        const size_t uRowStart = roff[u];
        const size_t uRowEnd = roff[u+1];


        // each thread follows edge u -> v from that vertex
        for (size_t uRowOff = uRowStart; uRowOff < uRowEnd; uRowOff += BLOCK_DIM) {
            size_t vIdx = uRowOff + threadIdx.x;
            int64_t v = vIdx < uRowEnd ? cols[vIdx] : -1; // -1 is sentinel for more threads than nodes adjacent to u

            if (v != -1) {
                const size_t vRowStart = roff[v];
                const size_t vRowEnd = roff[v+1];

                int local_counts = 0;
                local_counts = intersections(
                    &cols[uRowStart],
                    &cols[uRowEnd],
                    &cols[vRowStart],
                    &cols[vRowEnd]);
                if (local_counts) {
                    // printf("...in %lu -> %lu\n", u, v);
                }
                count += local_counts;
            }

        }
    }

    if (threadIdx.x == 0) {
        blockTriangleCounts[blockIdx.x] = 0;
    }
    __syncthreads();
    // block reduce
    atomicAdd(&blockTriangleCounts[blockIdx.x], count);

}

__global__ static void kernel_tc2(int * __restrict__ blockTriangleCounts, const Int *cols, const Int *roff, const size_t numRows){
     
    int count = 0;
    const size_t U_DST_BS = 32;
    __shared__ Int uDsts[U_DST_BS];

    // one block per vertex (u)
    for (size_t u = blockIdx.x; u < numRows; u += gridDim.x) {
        
        const size_t uRowStart = roff[u];
        const size_t uRowEnd = roff[u+1];

        // loop over chunks of vs in shared memory for the v's destinations to compare against i
        for (size_t uRowChunkOff = uRowStart; uRowChunkOff < uRowEnd; uRowChunkOff += U_DST_BS) {
            for (size_t i = threadIdx.x; i < U_DST_BS; i += BLOCK_DIM) {
                size_t idx = uRowChunkOff + i;
                if (idx < uRowEnd) {
                    uDsts[threadIdx.x] = cols[uRowChunkOff + i];
                } else {
                    uDsts[threadIdx.x] = -1;
                }
            }
            __syncthreads();


            // in each chunk of v's, follow all u-> edges in groups of BLOCK_DIM and compare against that chunk of v's
            // follow BLOCK_DIM u -> v edges at a time
            for (size_t uRowOff = uRowStart; uRowOff < uRowEnd; uRowOff += BLOCK_DIM) {

                size_t vIdx = uRowOff + threadIdx.x;
                int64_t v = vIdx < uRowEnd ? cols[vIdx] : -1; // -1 is sentinel for more threads than nodes adjacent to u

                if (v != -1) {
                    const size_t vRowStart = roff[v];
                    const size_t vRowEnd = roff[v+1];

                    int local_counts = 0;
                    local_counts = intersections(
                        &cols[vRowStart],
                        &cols[vRowEnd],
                        &uDsts[0],
                        &uDsts[U_DST_BS]);
                    if (local_counts) {
                        // printf("...in %lu -> %lu\n", u, v);
                    }
                    count += local_counts;
                }

            }


        }

    }

    if (threadIdx.x == 0) {
        blockTriangleCounts[blockIdx.x] = 0;
    }
    __syncthreads();
    // block reduce
    atomicAdd(&blockTriangleCounts[blockIdx.x], count);

}

VertexTC::VertexTC() {

    int numDev;
    CUDA_RUNTIME(cudaGetDeviceCount(&numDev));
    for (int i = 0; i < numDev; ++i) {
        LOG(info, "Initializing CUDA device {}", i);
        CUDA_RUNTIME(cudaSetDevice(i));
        CUDA_RUNTIME(cudaFree(0));
    }


    }

VertexTC::~VertexTC() {
}

void VertexTC::read_data(const std::string &path) {
        LOG(info, "reading {}", path);
        auto r = GraphChallengeTSVReader(path);
        const auto sz = r.size();
    
        auto edgeList = r.read_edges(0, sz);
        LOG(debug, "building DAG");
        dag_ = DAGLowerTriangularCSR::from_edgelist(edgeList);
    
        LOG(debug, "{} nodes", dag_.num_nodes());
        LOG(debug, "{} edges", dag_.num_edges());
    }

void VertexTC::setup_data() 
{

    const size_t srcBytes = dag_.sourceOffsets_.size() * sizeof(Int);
    const size_t dstBytes = dag_.destinationIndices_.size() * sizeof(Int);
    CUDA_RUNTIME(cudaMalloc((void **)&sourceOffsets_, srcBytes));
    CUDA_RUNTIME(cudaMalloc((void **)&destinationIndices_, dstBytes));
    CUDA_RUNTIME(cudaHostAlloc((void**)&blockTriangleCounts_, BLOCK_DIM * sizeof(int), cudaHostAllocMapped));
    CUDA_RUNTIME(cudaMemcpy(sourceOffsets_, dag_.sourceOffsets_.data(), srcBytes, cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpy(destinationIndices_, dag_.destinationIndices_.data(), dstBytes, cudaMemcpyDefault));
}

size_t VertexTC::count() {
    // LOG(warn, "incorrect results");
    uint64_t trcount = 0;

    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid(160);

    kernel_tc2<<<dimGrid, dimBlock>>>(blockTriangleCounts_, destinationIndices_, sourceOffsets_, dag_.num_nodes());
    CUDA_RUNTIME(cudaDeviceSynchronize());

    for (int i = 0; i < dimGrid.x; ++i) {
        trcount += blockTriangleCounts_[i];
    }

    return trcount;
}
