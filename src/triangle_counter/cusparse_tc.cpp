#include <memory>
#include <cmath>
#include <nvToolsExt.h>

#include "pangolin/logger.hpp"
#include "pangolin/triangle_counter/cusparse_tc.hpp"
#include "pangolin/reader/edge_list_reader.hpp"
#include "pangolin/utilities.hpp"


CUSparseTriangleCounter::CUSparseTriangleCounter(Config &c)
{
    if (c.gpus_.size() > 1)
    {
        gpu_ = c.gpus_[0];
        LOG(warn, "CUSparseTriangleCounter requires exactly 1 GPU. Selected GPU {}", gpu_);
    }
    else if (c.gpus_.size() == 0)
    {
        LOG(critical, "CUSparseTriangleCounter requires 1 GPU");
        exit(-1);
    }

    CUSPARSE(cusparseCreateHandle(&cusparseHandle_));

    int version;
    CUSPARSE(cusparseGetVersion(cusparseHandle, &version));
    LOG(info, "CUSparse version {}", version);
}

void CUSparseTriangleCounter::read_data(const std::string &path)
{
    LOG(info, "reading {}", path)
    auto *reader = pangolin::EdgeListReader::from_file(path);
    auto edgeList = reader->read();
    if (edgeList.size() == 0) {
        LOG(warn, "empty edge list");
    }
    LOG(debug, "building DAG");
    hostDAG_ = DAG2019::from_edgelist(edgeList);

    LOG(info, "reading {}", path);
    pangolin::GraphChallengeTSVReader r(path);
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

void CUSparseTriangleCounter::setup_data()
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
        for (Int o = rowStart; o < rowEnd; ++o)
        {
            TRACE("node {} off {} = {}", i, o, dag_.destinationIndices_[o]);
        }
    }
}

size_t CUSparseTriangleCounter::count()
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
