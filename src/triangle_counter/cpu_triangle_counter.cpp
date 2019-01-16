#if USE_OPENMP
#include <omp.h>
#endif

#include "pangolin/reader/gc_tsv_reader.hpp"
#include "pangolin/triangle_counter/cpu_triangle_counter.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/dag2019.hpp"

static size_t intersection_count(const Int *const aBegin, const Int *const aEnd, const Int *const bBegin, const Int *const bEnd)
{
    size_t count = 0;
    const Int *ap = aBegin;
    const Int *bp = bBegin;

    while (ap < aEnd && bp < bEnd)
    {

        if (*ap == *bp)
        {
            ++count;
            ++ap;
            ++bp;
        }
        else if (*ap < *bp)
        {
            ++ap;
        }
        else
        {
            ++bp;
        }
    }
    return count;
}

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
        GraphChallengeTSVReader r(path);
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

        Int u = dag_.edgeSrc_[i];
        Int v = dag_.edgeDst_[i];

        Int u_ptr = dag_.nodes_[u];
        Int u_end = dag_.nodes_[u + 1];

        Int v_ptr = dag_.nodes_[v];
        Int v_end = dag_.nodes_[v + 1];

        size_t count = intersection_count(&dag_.edgeDst_[u_ptr],
                                          &dag_.edgeDst_[u_end],
                                          &dag_.edgeDst_[v_ptr],
                                          &dag_.edgeDst_[v_end]);

#pragma omp atomic
        total += count;
    }

    return total;
}
