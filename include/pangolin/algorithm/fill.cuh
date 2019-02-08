#pragma once

PANGOLIN_NAMESPACE_BEGIN()

template<typename T>
__global__ void kernel_fill(T *a, const size_t n, const T val) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        a[i] = val;
    }
}

template <typename T>
void device_fill(T *d_a, const size_t n, const T val) {
    constexpr size_t dimBlockX = 256;
    const size_t dimGridX = (n + dimBlockX - 1) / dimBlockX;
    kernel_fill<<<dimGridX, dimBlockX>>>(d_a, n, val);
}

PANGOLIN_NAMESPACE_END()