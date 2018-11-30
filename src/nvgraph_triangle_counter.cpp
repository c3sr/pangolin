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

        auto numGroups = size_t(std::ceil(std::sqrt(numGPUs_)));
        LOG(debug, "creating {} groups", numGroups);

        // groups of random vertices
        std::vector<VertexSet> groups(numGroups);
        for (Int n = 0; n < dag.num_nodes(); ++n)
        {
            auto targetGroup = rand() % numGroups;
            LOG(trace, "node {} in group {}", n, targetGroup);
            groups[targetGroup].insert(n);
        }

        for (auto g : groups)
        {
            LOG(trace, "created a group with {} nodes", g.size());
        }

        dags_ = dag.partition(groups);
    }
    else
    {
        dags_.push_back(dag);
    }

    size_t partitionedTotalEdges = 0;
    size_t partitionedTotalNodes = 0;
    size_t partitionedMaxEdges = 0;
    size_t partitionedMaxNodes = 0;
    for (const auto &dag : dags_)
    {
        LOG(debug, "{} nodes", dag.num_nodes());
        LOG(debug, "{} edges", dag.num_edges());
        partitionedTotalEdges += dag.num_edges();
        partitionedTotalNodes += dag.num_nodes();
        partitionedMaxEdges = std::max(partitionedMaxEdges, dag.num_edges());
        partitionedMaxNodes = std::max(partitionedMaxNodes, dag.num_nodes());
    }

    LOG(info, "edge replication {}", double(partitionedTotalEdges) / dag.num_edges());
    LOG(info, "node replication {}", double(partitionedTotalNodes) / dag.num_nodes());
    LOG(info, "edge shrinkage {}", double(partitionedMaxEdges) / dag.num_edges());
    LOG(info, "node shrinkage {}", double(partitionedMaxNodes) / dag.num_nodes());
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