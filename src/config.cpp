#include "graph/config.hpp"
#include "graph/utilities.hpp"

Config::Config()
{
    int ndev;
    CUDA_RUNTIME(cudaGetDeviceCount(&ndev));
    for (size_t i = 0; i < ndev; ++i)
    {
        gpus_.push_back(i);
    }
    numCPUThreads_ = 1;
}