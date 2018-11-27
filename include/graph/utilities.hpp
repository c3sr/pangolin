#pragma once

#include <cuda_runtime.h>

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);

void checkCuda(cudaError_t result, const char *file, const int line);