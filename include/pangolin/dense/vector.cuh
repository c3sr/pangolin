#pragma once

#include "pangolin/allocator/cuda_managed.hpp"
#include "pangolin/logger.hpp"

#ifdef __CUDACC__
#define PANGOLIN_HOST_DEVICE __host__ __device__
#define PANGOLIN_HOST __host__
#else
#define PANGOLIN_HOST_DEVICE
#define PANGOLIN_HOST
#endif

#include <cstdlib>

namespace pangolin {
template <typename T, typename Allocator = allocator::CUDAManaged<T>> class Vector {
public:
  typedef T value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;

protected:
  size_t capacity_;  //<! the number of elements the underlying allocation can hold
  size_t size_;      //<! the number of elements in the vector
  value_type *data_; //<! the underlying allocation
  Allocator allocator_;

public:
  PANGOLIN_HOST explicit Vector(void) : capacity_(0), size_(0), data_(nullptr) {}

  PANGOLIN_HOST explicit Vector(size_t n) : capacity_(0), size_(0), data_(nullptr) { resize(n); }

  PANGOLIN_HOST explicit Vector(size_t n, const_reference val) : capacity_(0), size_(0), data_(nullptr) {
    resize(n, val);
  }

  PANGOLIN_HOST Vector(std::initializer_list<T> il) : capacity_(0), size_(0), data_(nullptr) {
    SPDLOG_TRACE(logger::console(), "il ctor");
    resize(il.size());
    auto it = il.begin();
    for (size_t i = 0; i < il.size(); ++i) {
      data_[i] = *it++;
    }
  }

  PANGOLIN_HOST Vector(Vector &&other) : capacity_(other.capacity_), size_(other.size_), data_(other.data_) {
    SPDLOG_TRACE(logger::console(), "move ctor");
    other.capacity_ = 0;
    other.size_ = 0;
    other.data_ = nullptr;
  }

  PANGOLIN_HOST Vector(const Vector &other) : capacity_(0), size_(0), data_(nullptr) {
    SPDLOG_TRACE(logger::console(), "copy ctor");
    reserve(other.capacity_);
    std::memcpy(data_, other.data_, other.size_ * sizeof(value_type));
    size_ = other.size_;
  }

  PANGOLIN_HOST ~Vector() {
    SPDLOG_TRACE(logger::console(), "dtor");
    if (data_) {
      for (size_t i = 0; i < size_; ++i) {
        (&data_[i])->~value_type();
      }

      allocator_.deallocate(data_, capacity_);
      data_ = nullptr;
      capacity_ = 0;
      size_ = 0;
    }
  }

  PANGOLIN_HOST Vector &operator=(Vector &&other) noexcept {
    SPDLOG_TRACE(logger::console(), "move assignment");
    /* We just swap other and this, which has the following benefits:
       Don't call delete on other (maybe faster)
       Opportunity for data to be reused since it was not deleted
       No exceptions thrown.
    */
    other.swap(*this);
    return *this;
  }

  PANGOLIN_HOST Vector &operator=(const Vector &other) {
    Vector<T> temp(other);
    temp.swap(*this);
    return *this;
  }

  PANGOLIN_HOST bool operator==(const Vector &other) const noexcept {
    if (size_ == other.size_) {
      for (size_t i = 0; i < size_; ++i) {
        if (data_[i] != other.data_[i]) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }

  PANGOLIN_HOST void swap(Vector &other) noexcept {
    std::swap(other.size_, size_);
    std::swap(other.capacity_, capacity_);
    std::swap(other.data_, data_);
    std::swap(other.allocator_, allocator_);
  }

  PANGOLIN_HOST_DEVICE inline value_type *data() noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline const value_type *data() const noexcept { return data_; }

  PANGOLIN_HOST void push_back(const value_type &val) {
    if (0 == capacity_) {
      reserve(1);
    } else if (size_ == capacity_) {
      reserve(capacity_ * 2);
    }
    data_[size_] = val;
    ++size_;
  }

  PANGOLIN_HOST void reserve(size_t n) {
    if (n > capacity_) {
      value_type *newData = allocator_.allocate(n);
      std::memmove(newData, data_, sizeof(value_type) * size_);
      allocator_.deallocate(data_, capacity_);
      data_ = newData;
      capacity_ = n;
    }
  }

  PANGOLIN_HOST void resize(size_t n) {
    if (n < size_) {
      // destroy elements beyond n
      for (size_t i = n; i < size_; ++i) {
        (&data_[i])->~value_type(); // manually call dtor
      }
      size_ = n;
    } else if (n > size_) {
      reserve(n);
      for (size_t i = size_; i < n; ++i) {
        new (&data_[i]) value_type(); // new elements are value-initialized
      }
      size_ = n;
    }
  }

  PANGOLIN_HOST void resize(size_t n, const value_type &val) {
    if (n < size_) {
      // destroy elements beyond n
      for (size_t i = n; i < size_; ++i) {
        (&data_[i])->~value_type(); // manually call dtor
      }
      size_ = n;
    } else if (n > size_) {
      reserve(n);
      for (size_t i = size_; i < n; ++i) {
        data_[i] = val;
      }
      size_ = n;
    }
  }

  PANGOLIN_HOST void shrink_to_fit() {
    if (capacity_ > size_) {
      value_type *newData = allocator_.allocate(size_);
      std::memmove(newData, data_, sizeof(value_type) * size_);
      allocator_.deallocate(data_, capacity_);
      data_ = newData;
      capacity_ = size_;
    }
  }

  PANGOLIN_HOST_DEVICE inline size_t capacity() const noexcept { return capacity_; }
  PANGOLIN_HOST_DEVICE inline size_t size() const noexcept { return size_; }
  PANGOLIN_HOST_DEVICE inline bool empty() const noexcept { return size_ == 0; }

  PANGOLIN_HOST_DEVICE inline reference operator[](size_t n) { return data_[n]; }
  PANGOLIN_HOST_DEVICE inline const_reference operator[](size_t n) const { return data_[n]; }

  PANGOLIN_HOST_DEVICE inline value_type *begin() noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline value_type *end() noexcept { return data_ + size_; }
  PANGOLIN_HOST_DEVICE inline const value_type *begin() const noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline const value_type *end() const noexcept { return data_ + size_; }

  /*! call cudaMemAdvise(..., cudaMemAdviseSetReadMostly, 0) on Vector data
   */
  PANGOLIN_HOST void read_mostly() {
    constexpr bool pred = std::is_same<Allocator, allocator::CUDAManaged<value_type>>::value;
    if (pred) {
      // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge37112fc1ac88d0f6bab7a945e48760a
      SPDLOG_TRACE(logger::console(), "cudaMemAdviseSetReadMostly {}B on device", size() * sizeof(T));
      CUDA_RUNTIME(cudaMemAdvise(data_, size() * sizeof(T), cudaMemAdviseSetReadMostly, 0 /* ignored */));
    }
  }

  /*! call cudaMemAdvise(..., cudaMemAdviseSetAccessedBy, dev) on Vector data
   */
  PANGOLIN_HOST void accessed_by(const int dev) {
    constexpr bool pred = std::is_same<Allocator, allocator::CUDAManaged<value_type>>::value;
    if (pred) {
      // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge37112fc1ac88d0f6bab7a945e48760a
      CUDA_RUNTIME(cudaMemAdvise(data_, size() * sizeof(T), cudaMemAdviseSetAccessedBy, dev));
    }
  }

  /*! call cudaMemPrefetchAsync(..., dev, stream) on Vector data
   */
  PANGOLIN_HOST void prefetch_async(const int dev, cudaStream_t stream = 0) {
    constexpr bool pred = std::is_same<Allocator, allocator::CUDAManaged<value_type>>::value;
    if (pred) {
      // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8dc9199943d421bc8bc7f473df12e42
      SPDLOG_TRACE(logger::console(), "cudaMemPrefetchAsync {}B to device {} stream {}", size() * sizeof(T), dev,
                   uintptr_t(stream));
      CUDA_RUNTIME(cudaMemPrefetchAsync(data_, size() * sizeof(T), dev, stream));
    }
  }
};

} // namespace pangolin

#undef PANGOLIN_HOST_DEVICE
#undef PANGOLIN_HOST
