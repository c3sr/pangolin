#pragma once

#include "graph/triangle_counter/cuda_triangle_counter.hpp"
#include "graph/dag2019.hpp"

#include <iostream>

class IMPACT2018TC : public CUDATriangleCounter
{

  public:
	enum class GPUMemoryKind
	{
		ZeroCopy,
		Unified
	};

  private:
	DAG2019 hostDAG_;
	size_t *triangleCounts_; // per-edge triangle counts
	Int *edgeSrc_d_;
	Int *edgeDst_d_;
	Int *nodes_d_;
	GPUMemoryKind GPUMemoryKind_;

  public:
	IMPACT2018TC(Config &c);
	virtual ~IMPACT2018TC();
	virtual void read_data(const std::string &path) override;
	virtual void setup_data() override;
	virtual size_t count() override;
	virtual size_t num_edges() override { return hostDAG_.num_edges(); }
	virtual size_t num_nodes() override { return hostDAG_.num_nodes(); }
};