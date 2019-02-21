#pragma once

#include "pangolin/triangle_counter/cuda_triangle_counter.hpp"
#include "pangolin/sparse/dag2019.hpp"
#include "pangolin/sparse/dag2019.hpp"

#include <iostream>

PANGOLIN_BEGIN_NAMESPACE()

/*! The 2019 IMPACT Triangle Counter

*/
class IMPACT2019TC : public CUDATriangleCounter
{

  public:
	enum class GPUMemoryKind
	{
		ZeroCopy,
		Unified
	};

	enum class KernelKind
	{
		Linear,
		Binary
	};


  private:
	DAG2019 hostDAG_;
	size_t *triangleCounts_; //!< per-edge triangle counts
	Int *edgeSrc_d_; //!< device edge srcs
	Int *edgeDst_d_; //!< device edge dsts
	Int *cols_d_; //!< device CSR column offsets in edgeDst_d_
	bool unifiedMemoryHints_; //!< whether to use unified memory hints
	GPUMemoryKind GPUMemoryKind_; //!< which kind of GPU memory to use to hold the data
	KernelKind kernelKind_; //!< which kernel to use to count triangles

  public:
	IMPACT2019TC(Config &c);
	virtual ~IMPACT2019TC();
	virtual void read_data(const std::string &path) override;
	virtual void setup_data() override;
	virtual size_t count() override;
	virtual uint64_t num_edges() override { return hostDAG_.num_edges(); }
	virtual size_t num_nodes() override { return hostDAG_.num_nodes(); }
};

PANGOLIN_END_NAMESPACE()