#pragma once

#include "graph/cuda_triangle_counter.hpp"
#include "graph/par_graph.hpp"

#include <vector>
#include <iostream>
#include <cuda_runtime.h>

class VertexTC : public CUDATriangleCounter
{
private:
	enum class CountingMethod
	{
		LINEAR,
		BINARY_TREE,
		HASH
	};

private:
	size_t numCPUThreads_; // how many threads to use

	// partitioned data structures
	std::vector<ParGraph> graphs_;

	// per-partition GPU data
	std::vector<Int *> rowStarts_d_;
	std::vector<Int *> nonZeros_d_;
	std::vector<bool *> isLocalNonZero_d_;
	std::vector<dim3> dimGrids_;
	// per-block triangle counts for each partition
	std::vector<size_t *> triangleCounts_;

	size_t numEdges_; // edges in input graph
	size_t numNodes_; // nodes in input graph

public:
	VertexTC(Config &c);
	virtual ~VertexTC();
	virtual void read_data(const std::string &path) override;
	virtual void setup_data() override;
	virtual size_t count() override;
	virtual size_t num_edges() override { return numEdges_; }
	virtual size_t num_nodes() { return numNodes_; }
};