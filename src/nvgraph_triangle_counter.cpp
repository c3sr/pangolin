#include "graph/nvgraph_triangle_counter.hpp"
#include "graph/logger.hpp"
#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/utilities.hpp"

void NvGraphTriangleCounter::read_data(const std::string &path)
{
    {
        LOG(info, "reading {}", path);
        auto r = GraphChallengeTSVReader(path);
        const auto sz = r.size();

        auto edgeList = r.read_edges(0, sz);
        LOG(debug, "building DAG");
        dag_ = DAGLowerTriangularCSR::from_edgelist(edgeList);

        LOG(debug, "{} nodes", dag_.num_nodes());
        LOG(debug, "{} edges", dag_.num_edges());
    }
}

void NvGraphTriangleCounter::setup_data()
{
    assert(sizeof(Int) == sizeof(int));
    csr_ = new struct nvgraphCSRTopology32I_st;
    csr_->nvertices = dag_.num_nodes();
    csr_->nedges = dag_.num_edges();

    Int *sourceOffsets;
    Int *destinationIndices;

    const size_t srcBytes = dag_.sourceOffsets_.size() * sizeof(Int);
    const size_t dstBytes = dag_.destinationIndices_.size() * sizeof(Int);
    CUDA_RUNTIME(cudaMalloc((void **)&sourceOffsets, srcBytes));
    CUDA_RUNTIME(cudaMalloc((void **)&destinationIndices, dstBytes));
    CUDA_RUNTIME(cudaMemcpy(sourceOffsets, dag_.sourceOffsets_.data(), srcBytes, cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpy(destinationIndices, dag_.destinationIndices_.data(), dstBytes, cudaMemcpyDefault));

    csr_->source_offsets = reinterpret_cast<int *>(sourceOffsets);
    csr_->destination_indices = reinterpret_cast<int *>(destinationIndices);
}

size_t NvGraphTriangleCounter::count()
{

    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graphDes;

    NVGRAPH(nvgraphCreate(&handle));
    NVGRAPH(nvgraphCreateGraphDescr(handle, &graphDes));

    NVGRAPH(nvgraphSetGraphStructure(handle, graphDes, (void *)csr_, NVGRAPH_CSR_32));

    uint64_t trcount = 0;
    NVGRAPH(nvgraphTriangleCount(handle, graphDes, &trcount));

    NVGRAPH(nvgraphDestroyGraphDescr(handle, graphDes));
    NVGRAPH(nvgraphDestroy(handle));

    return trcount;
}