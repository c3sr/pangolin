#include "graph/utilities.hpp"
#include "graph/logger.hpp"

void checkCuda(cudaError_t result, const char *file, const int line)
{
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error %s@%i: %s\n", file, line,
				cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
}

void checkNvgraph(nvgraphStatus_t result, const char *file, const int line)
{
	if (result != NVGRAPH_STATUS_SUCCESS)
	{
		printf("ERROR : %d in %s : %d\n", result, __FILE__, __LINE__);
		exit(0);
	}
}