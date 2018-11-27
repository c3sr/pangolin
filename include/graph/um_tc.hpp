#pragma once

#include "graph/triangle_counter.hpp"
#include "graph/dag2019.hpp"

#include <vector>
#include <iostream>

class UMTC : public TriangleCounter
{
private:
	DAG2019 hostDAG_;
	size_t *triangleCounts_; // per-edge triangle counts
	Int *edgeSrc_d_;
	Int *edgeDst_d_;
	Int *nodes_d_;

public:
	UMTC();
	virtual ~UMTC();
	virtual void read_data(const std::string &path) override;
	virtual void setup_data() override;
	virtual size_t count() override;
};