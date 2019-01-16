#pragma once

#include <cuda_runtime.h>
#include <nvgraph.h>

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line);

#define NVGRAPH(stmt) checkNvgraph(stmt, __FILE__, __LINE__);
void checkNvgraph(nvgraphStatus_t result, const char *file, const int line);
