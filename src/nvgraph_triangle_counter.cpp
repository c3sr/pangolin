#include <memory>
#include <cmath>

#include "graph/nvgraph_triangle_counter.hpp"
#include "graph/logger.hpp"
#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/utilities.hpp"

NvGraphTriangleCounter::NvGraphTriangleCounter(Config &c)
{
    numGPUs_ = c.numGPUs_;
    if (numGPUs_ == 0)
    {
        LOG(critical, "NvGraphTriangleCounter requires more than 0 GPUs");
        exit(-1);
    }
}

void NvGraphTriangleCounter::read_data(const std::string &path)
{
    LOG(info, "reading {}", path);
    auto r = GraphChallengeTSVReader(path);
    const auto sz = r.size();

    auto edgeList = r.read_edges(0, sz);
    LOG(debug, "building DAG");
    auto dag = DAGLowerTriangularCSR::from_edgelist(edgeList);

    LOG(debug, "{} nodes", dag.num_nodes());
    LOG(debug, "{} edges", dag.num_edges());

    if (numGPUs_ > 1)
    {
        LOG(debug, "partitioning for {} GPUs", numGPUs_);

        auto numGroups = std::ceil(std::sqrt(numGPUs_));
        size_t groupSize = (dag.num_nodes() + numGroups - 1) / numGroups;
        LOG(debug, "selected {} groups of size {}", numGroups, groupSize);

        std::vector<std::shared_ptr<VertexGroup>> groups;

        size_t rangeStart = 0;
        for (int i = 0; i < numGroups; ++i)
        {
            size_t rangeEnd = std::min(rangeStart + groupSize, dag.num_nodes());
            LOG(debug, "Adding group for vertices {} to {}", rangeStart, rangeEnd);
            groups.push_back(std::make_shared<VertexRange>(rangeStart, rangeEnd));
            rangeStart += groupSize;
        }

        std::vector<VertexGroup *> raw;
        for (auto g : groups)
        {
            raw.push_back(g.get());
        }

        dags_ = dag.partition(raw);
    }
    else
    {
        dags_.push_back(dag);
    }
    for (const auto &dag : dags_)
    {
        LOG(debug, "{} nodes", dag.num_nodes());
        LOG(debug, "{} edges", dag.num_edges());
    }
}

void NvGraphTriangleCounter::setup_data()
{
    assert(sizeof(Int) == sizeof(int));

    for (const auto &dag : dags_)
    {
        nvgraphCSRTopology32I_t csr = new struct nvgraphCSRTopology32I_st;
        csr->nvertices = dag.num_nodes();
        csr->nedges = dag.num_edges();

        const size_t srcBytes = dag.sourceOffsets_.size() * sizeof(Int);
        const size_t dstBytes = dag.destinationIndices_.size() * sizeof(Int);
        CUDA_RUNTIME(cudaMalloc((void **)&(csr->source_offsets), srcBytes));
        CUDA_RUNTIME(cudaMalloc((void **)&(csr->destination_indices), dstBytes));
        CUDA_RUNTIME(cudaMemcpy(csr->source_offsets, dag.sourceOffsets_.data(), srcBytes, cudaMemcpyDefault));
        CUDA_RUNTIME(cudaMemcpy(csr->destination_indices, dag.destinationIndices_.data(), dstBytes, cudaMemcpyDefault));

        csrs_.push_back(csr);

        LOG(trace, "dag with {} edges and {} nodes", csr->nedges, csr->nvertices);
        for (int i = 0; i < dag.num_nodes(); ++i)
        {
            Int rowStart = dag.sourceOffsets_[i];
            Int rowEnd = dag.sourceOffsets_[i + 1];
            for (size_t o = rowStart; o < rowEnd; ++o)
            {
                LOG(trace, "node {} off {} = {}", i, o, dag.destinationIndices_[o]);
            }
        }
    }
}

size_t NvGraphTriangleCounter::count()
{

    size_t trcount = 0;
    for (const auto &csr : csrs_)
    {

        if (csr->nvertices == 0)
        {
            continue;
        }

        LOG(debug, "nvgraph on nedges: {} nvertices: {}", csr->nedges, csr->nvertices);

        nvgraphHandle_t handle;
        nvgraphGraphDescr_t graphDes;

        NVGRAPH(nvgraphCreate(&handle));
        NVGRAPH(nvgraphCreateGraphDescr(handle, &graphDes));

        NVGRAPH(nvgraphSetGraphStructure(handle, graphDes, (void *)csr, NVGRAPH_CSR_32));

        uint64_t subgraph_triangle_count = 0;
        NVGRAPH(nvgraphTriangleCount(handle, graphDes, &subgraph_triangle_count));
        LOG(debug, "{} triangle", subgraph_triangle_count);

        NVGRAPH(nvgraphDestroyGraphDescr(handle, graphDes));
        NVGRAPH(nvgraphDestroy(handle));
        trcount += subgraph_triangle_count;
    }

    return trcount;
}