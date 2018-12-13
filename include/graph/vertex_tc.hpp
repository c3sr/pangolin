#pragma once

#include "graph/cuda_triangle_counter.hpp"
#include "graph/sparse/unified_memory_csr.hpp"
#include "graph/dense/cuda_managed_vector.hpp"

#include <vector>
#include <iostream>
#include <cuda_runtime.h>

class VertexTC : public CUDATriangleCounter
{
private:
	enum class Kernel
	{
		LINEAR,
		LINEAR_SHARED,
		BINARY,
		HASH
	};

private:
	Kernel kernel_;

	// partitioned data structures
	std::vector<UnifiedMemoryCSR> graphs_;

	// per-block triangle counts for each partition
	std::vector<CUDAManagedVector<uint64_t>> triangleCounts_;

	// per-partition device pointers
	std::vector<const Uint *> rowOffsets_d_;
	std::vector<const Uint *> nonZeros_d_;
	std::vector<const char *> isLocalNonZero_d_;
	std::vector<dim3> dimGrids_;
	std::vector<uint64_t *> triangleCounts_d_;

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