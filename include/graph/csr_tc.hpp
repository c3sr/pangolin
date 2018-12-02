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
	size_t numCPUThreads_;  // how many threads to use
	ParGraph graph_;
	Uint *triangleCounts_; // per-edge triangle counts
	Int *rowStarts_d_;
	Int *nonZeros_d_;
	bool *isLocalNonZero_d_;
	dim3 dimGrid_;

  public:
	CSRTC(Config &c);
	virtual ~CSRTC();
	virtual void read_data(const std::string &path) override;
	virtual void setup_data() override;
	virtual size_t count() override;
};