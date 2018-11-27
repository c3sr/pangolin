#if USE_OPENMP
#include <omp.h>
#endif

#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/cpu_triangle_counter.hpp"
#include "graph/logger.hpp"
#include "graph/dag2019.hpp"

CPUTriangleCounter::CPUTriangleCounter(const Config &c)
{
    numThreads_ = c.numCPUThreads_;
    LOG(debug, "config requested {} threads", numThreads_);

#if USE_OPENMP
    const size_t max_threads = omp_get_max_threads();
#else
    const size_t max_threads = 1;
#endif

    if (numThreads_ == 0)
    {
        numThreads_ = max_threads;
    }
    numThreads_ = std::min(max_threads, numThreads_);

#if USE_OPENMP
    omp_set_num_threads(numThreads_);
#endif

    LOG(info, "CPU Triangle Counter with {} threads", numThreads_);
}

void CPUTriangleCounter::read_data(const std::string &path)
{

    {
        LOG(info, "reading {}", path);
        auto r = GraphChallengeTSVReader(path);
        const auto sz = r.size();

        auto edgeList = r.read_edges(0, sz);
        LOG(debug, "building DAG");
        dag_ = DAG2019::from_edgelist(edgeList);
    }

    LOG(info, "{} nodes", dag_.num_nodes());
    LOG(info, "{} edges", dag_.num_edges());
}

size_t CPUTriangleCounter::count()
{

    size_t total = 0;

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < dag_.num_edges(); ++i)
    {

        size_t count = 0;

        Int u = dag_.edgeSrc_[i];
        Int v = dag_.edgeDst_[i];

        Int u_ptr = dag_.nodes_[u];
        Int u_end = dag_.nodes_[u + 1];

        Int v_ptr = dag_.nodes_[v];
        Int v_end = dag_.nodes_[v + 1];

        Int v_u, v_v;

        while ((u_ptr < u_end) && (v_ptr < v_end))
        {

            v_u = dag_.edgeDst_[u_ptr];
            v_v = dag_.edgeDst_[v_ptr];

            if (v_u == v_v)
            {
                ++count;
                ++u_ptr;
                ++v_ptr;
            }
            else if (v_u < v_v)
            {
                ++u_ptr;
            }
            else
            {
                ++v_ptr;
            }
        }

#pragma omp atomic
        total += count;
    }

    return total;
}
