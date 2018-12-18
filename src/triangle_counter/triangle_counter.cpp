#include "graph/triangle_counter/triangle_counter.hpp"
#include "graph/triangle_counter/cpu_triangle_counter.hpp"
#include "graph/triangle_counter/cudamemcpy_tc.hpp"
#include "graph/triangle_counter/impact_2018_tc.hpp"
#include "graph/triangle_counter/nvgraph_triangle_counter.hpp"
#include "graph/triangle_counter/hu_tc.hpp"
#include "graph/triangle_counter/vertex_tc.hpp"
#include "graph/triangle_counter/edge_tc.hpp"

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
    else if (c.type_ == "hu")
    {
        return new Hu2018TC(c);
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
        return new VertexTC(c);
    }
    else if (c.type_ == "edge")
    {
        return new EdgeTC(c);
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