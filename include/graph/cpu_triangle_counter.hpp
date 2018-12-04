#pragma once

#include "graph/triangle_counter.hpp"
#include "graph/dag2019.hpp"
#include "graph/config.hpp"

#include <vector>
#include <iostream>

class CPUTriangleCounter : public TriangleCounter
{
private:
	DAG2019 dag_;
	size_t numThreads_;

public:
	CPUTriangleCounter(const Config &c);
	virtual void read_data(const std::string &path) override;
	virtual size_t count() override;
	virtual size_t num_edges() override { return dag_.num_edges(); }
};