#pragma once

#ifdef __CUDACC__
#define PANGOLIN_HOST_DEVICE __host__ __device__
#define PANGOLIN_HOST __host__
#define PANGOLIN_DEVICE __device__
#else
#define PANGOLIN_HOST_DEVICE
#define PANGOLIN_HOST
#define PANGOLIN_DEVICE
#endif