#pragma once

#include "pangolin/triangle_counter/cuda_triangle_counter.hpp"
#include "pangolin/sparse/gpu_csr.hpp"

#include <iostream>

/*! The 2019 IMPACT Triangle Counter

*/
class SpmmTC : public CUDATriangleCounter
{
  private:
	uint64_t *edgeCnt_; //<! per-edge triangle counts
	Uint *edgeSrc_; //<! src of edge i
	Uint *edgeDst_; //<! dst of edge i
	pangolin::GPUCSR<Uint> aL_; //<! lower-triangular adjacency matrix
	pangolin::GPUCSR<Uint> aU_; //<! upper-triangular adjacency matrix
	uint64_t *nextEdge_; // next edge that GPU counter should write an edge count to

  public:
	SpmmTC(Config &c);
	virtual ~SpmmTC();
	virtual void read_data(const std::string &path) override;
	virtual void setup_data() override;
	virtual size_t count() override;
	virtual uint64_t num_edges() override { return aL_.nnz(); }
	virtual size_t num_nodes() override { return aL_.num_nodes(); }
};