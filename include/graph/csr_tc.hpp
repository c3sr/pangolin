#pragma once

#include "graph/triangle_counter.hpp"
#include "graph/par_graph.hpp"

#include <vector>
#include <iostream>
#include <cuda_runtime.h>

class CSRTC : public TriangleCounter
{
private:
	std::vector<int> gpus_; // which GPUs to use
	size_t numCPUThreads_;	// how many threads to use

	// partitioned data structures
	std::vector<ParGraph> graphs_;

	// per-block triangle counts
	std::vector<Uint *> triangleCounts_;
	std::vector<Int *> rowStarts_d_;
	std::vector<Int *> nonZeros_d_;
	std::vector<bool *> isLocalNonZero_d_;
	std::vector<dim3> dimGrids_;

public:
	CSRTC(Config &c);
	virtual ~CSRTC();
	virtual void read_data(const std::string &path) override;
	virtual void setup_data() override;
	virtual size_t count() override;
};