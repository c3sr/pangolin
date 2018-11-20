/* Author: Ketan Date */

#pragma once

#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/host_vector.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <ctime>
#include <set>

#define CSR_READ
#define SNAP_READ

#define NUMDEV 4
#define WORK_FRAC_GPU 0.9f
#define MAX_EDGES_GPU 0xFFFFFFFFLL

#define BLOCKDIMX 16
#define BLOCKDIMY 16

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);

void cudaMemoryMeasure(void);

void checkCuda(cudaError_t result, const char *file, const int line);

void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, long long int &total_blocks, long long int size);

void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, long long int &total_blocks, long long int size);
void printDeviceArray(long long int *d_array, long long int size);

void readGraph_SNAP_CSR(const char* filename, std::vector<int> &edge_vec_src, std::vector<int> &edge_vec_dest, std::vector<long long int> &row_ptrs, long long int &edgecount, int &nodecount);
void readGraph_DIMACS(const char *filename, std::vector<long long int> &edge_vec, long long int &edgecount, int &nodecount);
void readGraph_DARPA(const char *filename, std::vector<long long int> &edge_vec, long long int &edgecount, int &nodecount);

#ifndef CSR_READ
void readGraph_DARPA_CSR(const char* filename, std::vector<long long int> &edge_vec, std::vector<long long int> &row_ptrs, long long int &edgecount, int &nodecount);
#endif
#ifdef CSR_READ
void readGraph_DARPA_CSR(const char* filename, std::vector<int> &edge_vec_src, std::vector<int> &edge_vec_dest, std::vector<long long int> &row_ptrs, long long int &edgecount, int &nodecount);
#endif

void readGraph_DARPA_CSR_Full(const char* filename, std::vector<int> &edge_vec_src, std::vector<int> &edge_vec_dest, std::vector<long long int> &row_ptrs, long long int &edgecount, int &nodecount);

void removeDuplicateEdges(std::vector<long long int> &edge_vec, long long int &edgecount);
void removeSelfLoops(std::set<long long int> &edge_vec, long long int &edgecount);

void readEdgesFromBinFile(const char* filename, std::vector<long long int> &edge_vec, int &nodecount, long long int &edgecount);
int numVertices(std::vector<long long int> &edge_vec);

void printDeviceArray(int *d_array, long long int size);

inline __host__ __device__ long long int encodeEdge(int u, int v){
	return ((long long int)u) << 32 | v;
}

inline __host__ __device__ void decodeEdge(int &u, int &v, long long int e){
	u = (int)((e & 0xFFFFFFFF00000000LL) >> 32);
	v = (int)(e & 0xFFFFFFFFLL);
}

inline __host__ __device__ long long int binary_search(int* inarray, int value, long long int _min, long long int _max) {

	long long int min, mid, max;
	min = _min;
	max = _max;

	while (max - min > 1) {
		mid = min + ((max - min) / 2);
		int mid_val = inarray[mid];
		if (value == mid_val)
			return mid;
		else if (value > mid_val)
			min = mid;
		else
			max = mid;
	}

	int min_val = inarray[min];
	int max_val = inarray[max];

	if (value == min_val)
		return min;
	if (value == max_val)
		return max;

	return -1;
}

inline long long int __host__ __device__ get_min(long long int a, long long int b){
	if (a <= b)
		return a;
	return b;
}

inline long long int __host__ __device__ get_max(long long int a, long long int b){
	if (a >= b)
		return a;
	return b;
}

void writeGraph_DARPA(const char* filename, std::set<long long int> edge_set);
