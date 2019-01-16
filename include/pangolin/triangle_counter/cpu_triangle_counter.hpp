#pragma once

#include "pangolin/triangle_counter/triangle_counter.hpp"
#include "pangolin/sparse/dag2019.hpp"
#include "pangolin/config.hpp"

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