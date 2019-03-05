#pragma once

#ifdef __CUDACC__
#define PANGOLIN_HOST_DEVICE __host__ __device__
#else
#define PANGOLIN_HOST_DEVICE
#endif