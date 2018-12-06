#include "graph/triangle_counter.hpp"
#include "graph/cpu_triangle_counter.hpp"
#include "graph/csr_tc.hpp"
#include "graph/cudamemcpy_tc.hpp"
#include "graph/impact_2018_tc.hpp"
#include "graph/nvgraph_triangle_counter.hpp"
#include "graph/um_tc.hpp"
#include "graph/vertex_tc.hpp"
#include "graph/zc_tc.hpp"

TriangleCounter::~TriangleCounter() {}

void TriangleCounter::setup_data()
{
    LOG(debug, "triangle counter setup_data is a no-op");
}

TriangleCounter *TriangleCounter::CreateTriangleCounter(Config &c)
{
    if (c.type_ == "")
    {
        LOG(critical, "no counting method provided. Use -m flag");
        exit(-1);
    }
    else if (c.type_ == "impact")
    {
        return new IMPACT2018TC(c);
    }
    else if (c.type_ == "cudamemcpy")
    {
        return new CudaMemcpyTC();
    }
    else if (c.type_ == "nvgraph")
    {
        if (sizeof(Int) != sizeof(int))
        {
            LOG(critical, "nvgraph not supported for sizeof(Int) = {}", sizeof(Int));
            exit(-1);
        }
        return new NvGraphTriangleCounter(c);
    }
    else if (c.type_ == "vertex")
    {
        return new VertexTC();
    }
    else if (c.type_ == "csr")
    {
        return new CSRTC(c);
    }
    else if (c.type_ == "cpu")
    {
        return new CPUTriangleCounter(c);
    }
    else
    {
        LOG(critical, "unhandled triangle counter type: {}", c.type_);
        exit(-1);
    }
}