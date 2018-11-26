/* Author: Ketan Date */

#include "utilities.h"
#include "graph/logger.hpp"


void checkCuda(cudaError_t result, const char *file, const int line) {

	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error %s@%i: %s\n", file, line,
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
}

void cudaMemoryMeasure() {
	// Check cuda memory usage
	size_t free_byte;
	size_t total_byte;
	CUDA_RUNTIME(cudaMemGetInfo(&free_byte, &total_byte));
	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	std::cout << "GPU memory usage: used = " << used_db << " B, free = "
		<< free_db << " B, total = " << total_db <<
		" B" << std::endl;
}

// Function for calculating grid and block dimensions from the given input size.
void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, long long int &total_blocks, long long int size) {
	threads_per_block.x = 128;

	int value = size / (long long int) threads_per_block.x;
	if (size % (long long int) threads_per_block.x > 0)
		value++;

	total_blocks = value;
	blocks_per_grid.x = value;
}

// Function for calculating grid and block dimensions from the given input size for square grid.
void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, long long int &total_blocks, long long int size) {
	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;

	int sq_size = (int) ceil(sqrt(size)); 

	int valuex = (int) ceil((double) (sq_size) / BLOCKDIMX);
	int valuey = (int) ceil((double) (sq_size) / BLOCKDIMY);

	total_blocks = (long long int)valuex * (long long int)valuey;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
}

void printDeviceArray(long long int *d_array, long long int size){
	long long int *h_array = new long long int[size];
	cudaMemcpy(h_array, d_array, size * sizeof(long long int), cudaMemcpyDeviceToHost);
	for (long long int i = 0; i < size - 1; i++)
		std::cout << h_array[i] << "\t";
	std::cout << h_array[size - 1] << std::endl;
}

void printDeviceArray( int *d_array, long long int size){
	int *h_array = new int[size];
	cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
	for (long long int i = 0; i < size - 1; i++)
		std::cout << h_array[i] << "\t";
	std::cout << h_array[size - 1] << std::endl;
}

void readGraph_SNAP_CSR(const char* filename, std::vector<int> &edge_vec_src, std::vector<int> &edge_vec_dest, std::vector<long long int> &row_ptrs, long long int &edgecount, int &nodecount) {

	std::ifstream ss(filename);

	// count number of edges
	std::map<int, int> node_map;
	std::set<long long int> edge_set;

	int nodes = 0;

	int u, v;

	if (ss.is_open() && ss.good()){

		while (ss >> u){

				ss >> v;

				if (!node_map.count(u))
					node_map[u] = nodes++;
				if (!node_map.count(v))
					node_map[v] = nodes++;

				int uu = node_map[u];
				int vv = node_map[v];

				if (uu != vv) {
					int uuu = uu < vv ? uu : vv;
					int vvv = uu < vv ? vv : uu;
					edge_set.insert(encodeEdge(uuu, vvv));
				}
			}

		ss.close();
	}

	nodecount = node_map.size();
	edgecount = edge_set.size();

	int prevkey = -1;
	std::vector<std::pair<int, long long int> > temp_row_ptrs_vec;
	int edgecnt_tmp = 0;
	for (std::set<long long int>::iterator it = edge_set.begin(); it != edge_set.end(); ++it){
		int key, val;
		decodeEdge(key, val, *it);

		edge_vec_src.push_back(key);
		edge_vec_dest.push_back(val);

		nodecount = std::max<int>(nodecount, key);

		if (prevkey != key) {
			prevkey = key;
			temp_row_ptrs_vec.push_back(std::pair<int, long long int>(key, edgecnt_tmp));
		}
		edgecnt_tmp++;
	}

	edge_set.clear();

	long long int *temp_row_ptrs = new long long int[nodecount + 1];
	std::fill(temp_row_ptrs, temp_row_ptrs + nodecount, -1);
	temp_row_ptrs[nodecount] = edgecount;

	std::vector<std::pair<int, long long int> >::iterator begin = temp_row_ptrs_vec.begin();
	std::vector<std::pair<int, long long int> >::iterator end = temp_row_ptrs_vec.end();

	for (std::vector<std::pair<int, long long int> >::iterator it = begin; it != end; ++it)
		temp_row_ptrs[it->first] = it->second;

	long long int cur_val = edgecount;
	for (int i = nodecount; i >= 0; i--){
		long long int val = temp_row_ptrs[i];
		if (val < 0)
			temp_row_ptrs[i] = cur_val;
		else
			cur_val = val;
	}

	row_ptrs.insert(row_ptrs.begin(), temp_row_ptrs, temp_row_ptrs + nodecount + 1);

	delete[] temp_row_ptrs;
}

void readGraph_DIMACS(const char *filename, std::vector<long long int> &edge_vec, long long int &edgecount, int &nodecount){
	
	std::ifstream ss(filename);

	// count number of edges
	assert(ss.is_open());
	
	if (ss.is_open() && ss.good()){
		long long int n, m;
		std::string s;
		std::getline(ss, s);
		std::istringstream infile(s);
		infile >> n;
		infile >> m;

		for (int u = 0; u < n; ++u) {
			std::getline(ss, s);
			std::istringstream parser(s);
			int v;

			while (parser >> v)
				edge_vec.push_back(encodeEdge(u, v - 1));
		}

		ss.close();

		nodecount = n;
	}

	edgecount = edge_vec.size();

}


void removeDuplicateEdges(std::vector<long long int> &edge_vec, long long int &edgecount){
	std::sort(edge_vec.begin(), edge_vec.end());
	edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());

	edgecount = edge_vec.size();
}
/*
void removeSelfLoops(std::set<long long int> &edge_vec, long long int &edgecount) {

	edge_vec.erase(std::remove_if(edge_vec.begin(), edge_vec.end(),
		[&](long long int e) { int u, v; decodeEdge(u, v, e);  return u == v; }), edge_vec.end());

	edgecount = edge_vec.size();
}
*/
void readEdgesFromBinFile(const char* filename, std::vector<long long int> &edge_vec, int &nodecount, long long int &edgecount) {
	std::ifstream in(filename, std::ios::binary);
	
	in.read((char*)&nodecount, sizeof(int));
	in.read((char*)&edgecount, sizeof(long long int));
	edge_vec.resize(edgecount);
	in.read((char*)edge_vec.data(), edgecount * sizeof(long long int));
}

int numVertices(std::vector<long long int> &edge_vec) {
	int num_vertices = 0;
	for (const long long int e : edge_vec) {
		int u, v;
		decodeEdge(u, v, e);
		num_vertices = std::max(num_vertices, 1 + std::max(u, v));
	}
	return num_vertices;
}

void readGraph_DARPA(const char* filename, std::vector<long long int> &edge_vec, long long int &edgecount, int &nodecount) {
	std::ifstream ss(filename);


	edgecount = 0;
	nodecount = 0;

	int val, key, weight;

	if (ss.is_open() && ss.good()){
		while (ss >> val){
			ss >> key;
			ss >> weight;

			nodecount = std::max<int>(nodecount, key);

			key--;
			val--;

			long long int edge = encodeEdge(key, val);
			edge_vec.push_back(edge);
		}

		ss.close();
	}

	edgecount = edge_vec.size();
}

#ifndef CSR_READ
void readGraph_DARPA_CSR(const char* filename, std::vector<long long int> &edge_vec, std::vector<long long int> &row_ptrs, long long int &edgecount, int &nodecount) {
#endif
#ifdef CSR_READ
void readGraph_DARPA_CSR(const char* filename, std::vector<int> &edge_vec_src,std::vector<int> &edge_vec_dest, std::vector<long long int> &row_ptrs, long long int &edgecount, int &nodecount) {
#endif




	int key, val, weight;
	std::ifstream ss(filename);
	std::vector<std::pair<int, long long int> > temp_row_ptrs_vec;

	if (!ss.good()) {
		LOG(critical, "couldn't open {}", filename);
		exit(-1);
	}

	edgecount = 0;
	nodecount = 0;
	int prevkey = -1;

	if (ss.is_open() && ss.good()){
		while (ss >> val){
				
			ss >> key;
			ss >> weight;

			nodecount = std::max<int>(nodecount, key);

			key--;
			val--;
	
			if(key < val) {
#ifndef CSR_READ
				long long int edge = encodeEdge(key, val);
#endif
#ifdef CSR_READ
				edge_vec_src.push_back(key);
				edge_vec_dest.push_back(val);
#endif
					
				if(prevkey != key) {
					prevkey = key;
					temp_row_ptrs_vec.push_back(std::pair<int, long long int>(key, edgecount));
				}

				edgecount++;
			}
		}

		ss.close();
	}

	long long int *temp_row_ptrs = new long long int[nodecount + 1];
	std::fill(temp_row_ptrs, temp_row_ptrs + nodecount, -1);
	temp_row_ptrs[nodecount] = edgecount;

	std::vector<std::pair<int, long long int> >::iterator begin = temp_row_ptrs_vec.begin();
	std::vector<std::pair<int, long long int> >::iterator end = temp_row_ptrs_vec.end();

	for(std::vector<std::pair<int, long long int> >::iterator it = begin; it != end; ++it) 
		temp_row_ptrs[it->first] = it->second;
	
	long long int cur_val = edgecount;
	for (int i = nodecount; i >= 0; i--){
		long long int val = temp_row_ptrs[i];
		if (val < 0)
			temp_row_ptrs[i] = cur_val;
		else
			cur_val = val;
	}

	row_ptrs.insert(row_ptrs.begin(), temp_row_ptrs, temp_row_ptrs + nodecount + 1);

	delete[] temp_row_ptrs;
}


void readGraph_DARPA_CSR_Full(const char* filename, std::vector<int> &edge_vec_src, std::vector<int> &edge_vec_dest, std::vector<long long int> &row_ptrs, long long int &edgecount, int &nodecount) {

	std::ifstream ss(filename);
	std::vector<std::pair<int, long long int> > temp_row_ptrs_vec;

	edgecount = 0;
	nodecount = 0;
	int prevkey = -1;

	int key, val, weight;

	if (ss.is_open() && ss.good()){
		while (ss >> val){
	
			ss >> key;
			ss >> weight;

				nodecount = std::max<int>(nodecount, key);

				key--;
				val--;

				edge_vec_src.push_back(key);
				edge_vec_dest.push_back(val);

				if (prevkey != key) {
					prevkey = key;
					temp_row_ptrs_vec.push_back(std::pair<int, long long int>(key, edgecount));
				}

				edgecount++;
				
			}
		

		ss.close();
	}

	long long int *temp_row_ptrs = new long long int[nodecount + 1];
	std::fill(temp_row_ptrs, temp_row_ptrs + nodecount, -1);
	temp_row_ptrs[nodecount] = edgecount;

	std::vector<std::pair<int, long long int> >::iterator begin = temp_row_ptrs_vec.begin();
	std::vector<std::pair<int, long long int> >::iterator end = temp_row_ptrs_vec.end();

	for (std::vector<std::pair<int, long long int> >::iterator it = begin; it != end; ++it)
		temp_row_ptrs[it->first] = it->second;

	long long int cur_val = edgecount;
	for (int i = nodecount; i >= 0; i--){
		long long int val = temp_row_ptrs[i];
		if (val < 0)
			temp_row_ptrs[i] = cur_val;
		else
			cur_val = val;
	}

	row_ptrs.insert(row_ptrs.begin(), temp_row_ptrs, temp_row_ptrs + nodecount + 1);

	delete[] temp_row_ptrs;
}

void writeGraph_DARPA(const char* filename, std::set<long long int> edge_set){
	std::stringstream ss;
	ss << filename << ".tsv";

	std::ofstream out(ss.str().c_str());

	for (std::set<long long int>::iterator it = edge_set.begin(); it != edge_set.end(); ++it) {

		int u, v;
		decodeEdge(u, v, *it);

		out << v << "\t" << u << "\t" << 1 << std::endl;

	}

	out.close();
}
