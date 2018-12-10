#pragma once

#include <cstring>

template <typename VALUE_TYPE>
class CUDAManagedVector
{
  public:
    typedef VALUE_TYPE value_type;
    typedef value_type &reference;
    typedef const value_type &const_reference;

  private:
    size_t capacity_;
    size_t size_;
    value_type *data_;

  public:
    explicit CUDAManagedVector() : capacity_(0), size_(0), data_(nullptr)
    {
    }

    ~CUDAManagedVector()
    {
        if (data_)
        {
            cudaFree(data_);
            data_ = nullptr;
        }
    }

    explicit CUDAManagedVector(size_t n)
    {
        reserve(n);
    }

    value_type *data() noexcept
    {
        return data_;
    }
    const value_type *data() const noexcept
    {
        return data_;
    }

    void push_back(const value_type &val)
    {
        if (size_ == capacity_)
        {
            reserve(capacity_ * 2);
        }
        data_[size_] = val;
        ++size_;
    }

    void reserve(size_t n)
    {
        if (n > capacity_)
        {
            value_type *newData = nullptr;
            cudaMallocManaged(&newData, sizeof(value_type) * n);
            if (!empty())
            {
                std::memmove(newData, data_, sizeof(value_type) * size_);
                cudaFree(data_);
                data_ = newData;
                capacity_ = n;
            }
        }
    }

    size_t capacity() const
    {
        return capacity_;
    }

    size_t size() const
    {
        return size_;
    }

    bool empty() const
    {
        return size() != 0;
    }

    reference operator[](size_t n)
    {
        return data_[n];
    }
    const_reference operator[](size_t n) const
    {
        return data_[n];
    }
};