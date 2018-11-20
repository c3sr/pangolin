#pragma once

#include "graph/triangle_counter.hpp"

#include <vector>
#include <iostream>

#define NUMDEV 1

class GPUTriangleCounter : public TriangleCounter

{



	long long int edge_split[NUMDEV + 2]; 
	long long int *d_rowptrs_dev[NUMDEV];

	int nodecount;
	long long int edgecount, working_edgecount;

	std::vector<long long int> edge_vec, row_ptrs_vec;
	std::vector<int> edge_vec_src, edge_vec_dest;

	long long int *cpu_tc;

	struct AdjList{
		long long int *edgeids;
		long long int *rowptrs;
		int *edgeids_src;
		int *edgeids_dest;
	} graph;

	long long int *working_edgelist;
	

public:

	GPUTriangleCounter();
	void execute(const char* filename, int omp_numthreads);

private:

	void computeWorkFractions(long long int *edge_split, long long int size);
	void calcRowPtrs(void);
	void prepAdj(void);
	void prepWorkingEdgeList(void);
	void edgeListByPriority(void);

	long long int countTriangles(void);
	void calculateWorkSplit(long long int count);
	void allocArrays(void);
	void freeArrays(void);

};