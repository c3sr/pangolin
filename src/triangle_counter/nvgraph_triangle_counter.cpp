#include <memory>
#include <cmath>

#include "graph/logger.hpp"
#include "graph/triangle_counter/nvgraph_triangle_counter.hpp"
#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/utilities.hpp"

#include <nvToolsExt.h>

NvGraphTriangleCounter::NvGraphTriangleCounter(Config &c)
{
    if (c.gpus_.size() > 1)
    {
        gpu_ = c.gpus_[0];
        LOG(warn, "NvGraphTriangleCounter requires exactly 1 GPU. Selected GPU {}", gpu_);
    }
    else if (c.gpus_.size() == 0)
    {
        LOG(critical, "NvGraphTriangleCounter requires 1 GPU");
        exit(-1);
    }
}

void NvGraphTriangleCounter::read_data(const std::string &path)
{
    LOG(info, "reading {}", path);
    GraphChallengeTSVReader r(path);
    const auto sz = r.size();

    auto edgeList = r.read_edges(0, sz);
    LOG(debug, "building DAG");
    dag_ = DAGLowerTriangularCSR::from_edgelist(edgeList);

    LOG(debug, "{} nodes", dag_.num_nodes());
    LOG(debug, "{} edges", dag_.num_edges());

    csr_ = new struct nvgraphCSRTopology32I_st;
    csr_->nvertices = dag_.num_nodes();
    csr_->nedges = dag_.num_edges();
}

void NvGraphTriangleCounter::setup_data()
{
    assert(sizeof(Int) == sizeof(int));

    const size_t srcBytes = dag_.sourceOffsets_.size() * sizeof(Int);
    const size_t dstBytes = dag_.destinationIndices_.size() * sizeof(Int);
    CUDA_RUNTIME(cudaMalloc((void **)&(csr_->source_offsets), srcBytes));
    CUDA_RUNTIME(cudaMalloc((void **)&(csr_->destination_indices), dstBytes));
    CUDA_RUNTIME(cudaMemcpy(csr_->source_offsets, dag_.sourceOffsets_.data(), srcBytes, cudaMemcpyDefault));
    CUDA_RUNTIME(cudaMemcpy(csr_->destination_indices, dag_.destinationIndices_.data(), dstBytes, cudaMemcpyDefault));

    TRACE("dag with {} edges and {} nodes", csr_->nedges, csr_->nvertices);
    for (size_t i = 0; i < dag_.num_nodes(); ++i)
    {
        Int rowStart = dag_.sourceOffsets_[i];
        Int rowEnd = dag_.sourceOffsets_[i + 1];
        for (size_t o = rowStart; o < rowEnd; ++o)
        {
            TRACE("node {} off {} = {}", i, o, dag_.destinationIndices_[o]);
        }
    }
}

size_t NvGraphTriangleCounter::count()
{

    if (csr_->nvertices == 0)
    {
        return 0;
    }
    uint64_t trcount = 0;

    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graphDes;

    NVGRAPH(nvgraphCreate(&handle));
    NVGRAPH(nvgraphCreateGraphDescr(handle, &graphDes));

    NVGRAPH(nvgraphSetGraphStructure(handle, graphDes, (void *)csr_, NVGRAPH_CSR_32));

    NVGRAPH(nvgraphTriangleCount(handle, graphDes, &trcount));

    NVGRAPH(nvgraphDestroyGraphDescr(handle, graphDes));
    NVGRAPH(nvgraphDestroy(handle));

    return trcount;
}
