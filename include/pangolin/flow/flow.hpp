#pragma once

#if USE_NUMA == 1
#include <numa.h>
#endif

#include "component.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {

template <typename T, size_t MAX_COMPONENTS = 16> class FlowVector {

public:
  typedef T value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;

private:
  cudaStream_t stream_;
  size_t numConsumers_;
  size_t numProducers_;
  Component producers_[MAX_COMPONENTS];
  Component consumers_[MAX_COMPONENTS];

  T *data_;
  size_t size_;
  size_t capacity_;

  std::function<void(T *)> deallocate_;
  std::function<T *(size_t n)> allocate_;

public:
  FlowVector() : stream_(0), numProducers_(0), numConsumers_(0), data_(nullptr), size_(0), capacity_(0) {
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
  }

  ~FlowVector() {
    if (stream_) {
      CUDA_RUNTIME(cudaStreamDestroy(stream_));
    }
  }

  FlowVector &add_producer(const Component &c) {
    assert(numProducers_ < MAX_COMPONENTS);
    producers_[numProducers_++] = c;
    return *this;
  }

  FlowVector &add_consumer(const Component &c) {
    assert(numConsumers_ < MAX_COMPONENTS);
    consumers_[numConsumers_++] = c;
    return *this;
  }

  void to_producer_async() {}
  void to_consumer_async() {}

  void to_consumer() {
    to_consumer_async();
    sync();
  }

  void to_producer() {
    to_producer_async();
    sync();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  T *data() { return data_; }
  const T *data() const { return data_; }

  void reserve(size_t n);

private:
  // NUMA allocator
  T *allocate_cpu_numa(size_t n) { return new T[n]; }
  void deallocate_numa(T *p) { delete[] p; }

  // CUDA unified memory allocator
  T *allocate_cuda(size_t n) {
    T *ptr;
    CUDA_RUNTIME(cudaMallocManaged(&ptr, sizeof(T) * n));
    return ptr;
  }
  void deallocate_cuda(T *p) { CUDA_RUNTIME(cudaFree(p)); }
};

} // namespace pangolin

#include "flow-impl.hpp"