/* Author: Ketan Date 
           Vikram Sharma Mailthdoy
 */

#include "utilities.h"
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/host_vector.h>


class CuTriangleCounter
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

#ifdef CSR_READ
__global__ void kernel_triangleCounter_tc(long long int *cpu_tc, int *cpu_edgeids_src, int *cpu_edgeids_dest, long long int *cpu_rowptrs, long long int size, long long int offset);
#endif
#ifndef CSR_READ
__global__ void kernel_triangleCounter_tc(long long int *cpu_tc, long long int *working_edgelist, long int *cpu_edgeids, long long int *cpu_rowptrs, long long int size, long long int offset);
#endif
