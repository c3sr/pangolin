#include "pangolin/utilities.hpp"
#include "pangolin/logger.hpp"

void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    LOG(critical, "{}@{}: CUDA Runtime Error: {}\n", file, line,
        cudaGetErrorString(result));
    exit(-1);
  }
}

void checkNvgraph(nvgraphStatus_t result, const char *file, const int line) {
  if (result != NVGRAPH_STATUS_SUCCESS) {
    printf("nvgraph Error: %s in %s : %d\n", nvgraphStatusGetString(result),
           file, line);
    exit(-1);
  }
}
