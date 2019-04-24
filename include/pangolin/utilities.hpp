#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <nvgraph.h>
#include <nvml.h>

#include "logger.hpp"

inline void checkNvml(nvmlReturn_t result, const char *file, const int line) {
  if (result != NVML_SUCCESS) {
    LOG(critical, "nvml Error: {} in {} : {}", nvmlErrorString(result), file, line);
    exit(-1);
  }
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
void checkCuda(cudaError_t result, const char *file, const int line);

#define NVGRAPH(stmt) checkNvgraph(stmt, __FILE__, __LINE__);
void checkNvgraph(nvgraphStatus_t result, const char *file, const int line);

#define NVML(stmt) checkNvml(stmt, __FILE__, __LINE__);
void checkNvgraph(nvmlReturn_t result, const char *file, const int line);
