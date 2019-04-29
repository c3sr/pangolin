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

inline void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    LOG(critical, "{}@{}: CUDA Runtime Error: {}\n", file, line,
        cudaGetErrorString(result));
    exit(-1);
  }
}

inline void checkNvgraph(nvgraphStatus_t result, const char *file, const int line) {
  if (result != NVGRAPH_STATUS_SUCCESS) {
    printf("nvgraph Error: %s in %s : %d\n", nvgraphStatusGetString(result),
           file, line);
    exit(-1);
  }
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
#define NVGRAPH(stmt) checkNvgraph(stmt, __FILE__, __LINE__);
#define NVML(stmt) checkNvml(stmt, __FILE__, __LINE__);
