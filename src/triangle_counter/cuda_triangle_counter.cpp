#include <nvToolsExt.h>

#include "pangolin/triangle_counter/cuda_triangle_counter.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/types.hpp"
#include "pangolin/utilities.hpp"

CUDATriangleCounter::CUDATriangleCounter(Config &c) : gpus_(c.gpus_), unique_gpus_(gpus_.begin(), gpus_.end())
{
    nvtxRangePush(__PRETTY_FUNCTION__);
    LOG(debug, "CUDA Triangle Counter, sizeof(Int) = {}", sizeof(Int));

    if (gpus_.empty())
    {
        LOG(critical, "CUDA Triangle Counter requires >= 1 GPU");
        exit(-1);
    }

    for (int dev : unique_gpus_)
    {
        LOG(info, "Initializing CUDA device {}", dev);
        CUDA_RUNTIME(cudaSetDevice(dev));
        CUDA_RUNTIME(cudaFree(0));
        if (0 == cudaDeviceProps_.count(dev))
        {
            CUDA_RUNTIME(cudaGetDeviceProperties(&cudaDeviceProps_[dev], dev));
        }
    }
    nvtxRangePop();
}
