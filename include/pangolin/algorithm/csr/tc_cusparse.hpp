#pragma once

#include <cmath>
#include <cub/cub.cuh>
#include <memory>
#include <nvToolsExt.h>

#include "../elementwise.cuh"
#include "../fill.cuh"
#include "pangolin/cusparse.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {

/*! Count triangles with CUSparse

Count triangle for a lower-triangular matrix A with (A x A .* A).

A x A = C
C .*= A

*/
class CUSparseTC {
private:
  int dev_;
  cudaStream_t stream_;

  cusparseHandle_t handle_;
  cusparseMatDescr_t descrA_;
  cusparseMatDescr_t descrC_;

  int *csrRowPtrC_; //!< Row pointer array for C
  int *csrColIndC_; //!< Column Index array for C
  float *csrValA_;  //!< Array of 1s for A
  float *csrValC_;  //!< Value array for C
  int nnzC_;        //!< nnc for C (<0 means unset)

  size_t tempStorageBytes_;
  void *dTempStorage_;
  float *deviceTotal_;

public:
  CUSparseTC(int dev, cudaStream_t stream = 0)
      : dev_(dev), stream_(stream), handle_(nullptr), descrA_(nullptr), descrC_(nullptr), csrRowPtrC_(nullptr),
        csrColIndC_(nullptr), csrValA_(nullptr), csrValC_(nullptr), nnzC_(-1), tempStorageBytes_(0),
        dTempStorage_(nullptr), deviceTotal_(nullptr) {
    LOG(debug, "create CUSparse handle");
    CUSPARSE(cusparseCreate(&handle_));
    CUSPARSE(cusparseSetStream(handle_, stream_));

    int version;
    CUSPARSE(cusparseGetVersion(handle_, &version));
    LOG(info, "CUSparse version {}", version);

    CUSPARSE(cusparseCreateMatDescr(&descrA_));
    CUSPARSE(cusparseCreateMatDescr(&descrC_));
  }

  CUSparseTC() : CUSparseTC(-1) {}

  template <typename CSR> void preallocate_row_ptr_c(const CSR &csr) {
    if (nullptr == csrRowPtrC_) {
      const int m = csr.num_rows();
      LOG(debug, "allocate {} rows for C", m);
      CUDA_RUNTIME(cudaMallocManaged(&csrRowPtrC_, sizeof(int) * (m + 1)));
    }
  }

  template <typename CSR> void precompute_layout_c(const CSR &csr) {

    if (nullptr == csrRowPtrC_) {
      preallocate_row_ptr_c(csr);
    }
    assert(csrRowPtrC_);

    const int m = csr.num_rows();
    const int n = csr.num_cols();
    const int k = csr.num_cols();
    const cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const int nnzA = csr.nnz();
    const int *csrRowPtrA = csr.row_ptr();
    const int *csrColIndA = csr.col_ind();
    LOG(debug, "compute C nnzs and set C rowPtrs");
    int *nnzTotalDevHostPtr = &nnzC_;
    CUSPARSE(cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST));
    CUSPARSE(cusparseXcsrgemmNnz(handle_, transA, transA, m, n, k, descrA_, nnzA, csrRowPtrA, csrColIndA, descrA_, nnzA,
                                 csrRowPtrA, csrColIndA, descrC_, csrRowPtrC_, nnzTotalDevHostPtr));
    if (nullptr != nnzTotalDevHostPtr) {
      SPDLOG_TRACE(logger::console(), "get nnzC from nnzTotalDevHostPtr");
      nnzC_ = *nnzTotalDevHostPtr;
      // assert(nnzC_ == csrRowPtrC_[m] - csrRowPtrC_[0]);
    } else {
      int baseC;
      nnzC_ = csrRowPtrC_[m];
      baseC = csrRowPtrC_[0];
      nnzC_ -= baseC;
    }
    LOG(debug, "C will have {} nonzeros", nnzC_);

    if (nullptr == csrColIndC_) {
      LOG(debug, "allocate {}B for csrColIndC", sizeof(int) * nnzC_);
      CUDA_RUNTIME(cudaMallocManaged(&csrColIndC_, sizeof(int) * nnzC_));
    }
    if (nullptr == csrValC_) {
      LOG(debug, "allocate {}B for csrValC", sizeof(float) * nnzC_);
      CUDA_RUNTIME(cudaMallocManaged(&csrValC_, sizeof(float) * nnzC_));
    }
  }

  template <typename CSR> void preallocate_vals_a(const CSR &csr) {
    const int nnzA = csr.nnz();
    if (nullptr == csrValA_) {
      LOG(debug, "allocate/fill {}B for csrValA", sizeof(float) * nnzA);
      CUDA_RUNTIME(cudaMallocManaged(&csrValA_, sizeof(float) * nnzA));
      pangolin::device_fill(csrValA_, nnzA, 1.0f);
      CUDA_RUNTIME(cudaDeviceSynchronize());
    }
  }

  template <typename CSR> void preallocate_reduction(const CSR &csr) {
    if (nullptr == deviceTotal_) {
      LOG(debug, "allocate/fill {}B for reduction total", sizeof(*deviceTotal_));
      CUDA_RUNTIME(cudaMallocManaged(&deviceTotal_, sizeof(*deviceTotal_)));
      *deviceTotal_ = 0;
    }

    if (nullptr == dTempStorage_) {
      if (nnzC_ < 0) {
        precompute_layout_c(csr);
      }
      assert(nnzC_ >= 0);
      tempStorageBytes_ = 0;
      LOG(debug, "compute reduction storage requirements");
      cub::DeviceReduce::Sum(dTempStorage_, tempStorageBytes_, csrValC_, deviceTotal_, nnzC_);
      LOG(debug, "allocate {} B for temporary reduction storage", tempStorageBytes_);
      CUDA_RUNTIME(cudaMalloc(&dTempStorage_, tempStorageBytes_));
    }
  }

  /*!
  Use CUSparse to count triangles in `csr`, allocating intermediate storage as needed.

  To pre-allocate intermediate storage, call some/all of
    preallocate_vals_a(csr);
    preallocate_row_ptr_c(csr);
    precompute_layout_c(csr);
    preallocate_reduction(csr);
  first

  */
  template <typename CSR> void count_async(const CSR &csr) {
    preallocate_vals_a(csr);
    preallocate_row_ptr_c(csr);
    precompute_layout_c(csr);
    preallocate_reduction(csr);

    const int m = csr.num_rows();
    const int n = csr.num_cols();
    const int k = csr.num_cols();
    LOG(debug, "CUSparse product m={} n={} k={}", m, n, k);

    const cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const int nnzA = csr.nnz();
    const int *csrRowPtrA = csr.row_ptr();
    const int *csrColIndA = csr.col_ind();

    assert(nnzA == csrRowPtrA[m] - csrRowPtrA[0]);
    LOG(debug, "A has {} nonzeros", nnzA);

    LOG(debug, "cusparseScsrgemm");
    // c = A * A
    CUSPARSE(cusparseScsrgemm(handle_, transA, transA, m, n, k, descrA_, nnzA, csrValA_, csrRowPtrA, csrColIndA,
                              descrA_, nnzA, csrValA_, csrRowPtrA, csrColIndA, descrC_, csrValC_, csrRowPtrC_,
                              csrColIndC_));

    LOG(debug, "hadamard product");
    // c .*= A
    constexpr size_t dimBlockX = 256;
    const size_t dimGridX = (m + dimBlockX - 1) / dimBlockX;
    assert(csrRowPtrC_);
    assert(csrColIndC_);
    assert(csrValC_);
    assert(csrValA_);
    pangolin::csr_elementwise_inplace<dimBlockX>
        <<<dimGridX, dimBlockX, 0, stream_>>>(csrRowPtrC_, csrColIndC_, csrValC_, csrRowPtrA, csrColIndA, csrValA_, m);
    CUDA_RUNTIME(cudaGetLastError());

    // Reduce the final non-zeros
    LOG(debug, "device reduction");
    assert(dTempStorage_);
    assert(deviceTotal_);
    cub::DeviceReduce::Sum(dTempStorage_, tempStorageBytes_, csrValC_, deviceTotal_, nnzC_, stream_);
  }

  template <typename CSR> uint64_t count_sync(const CSR &csr) {
    count_async(csr);
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
  uint64_t count() const { return *deviceTotal_; }
  int device() const { return dev_; }

  ~CUSparseTC() {
    LOG(debug, "destroy A");
    CUSPARSE(cusparseDestroyMatDescr(descrA_));
    LOG(debug, "destroy C");
    CUSPARSE(cusparseDestroyMatDescr(descrC_));
    LOG(debug, "destroy handle");
    CUSPARSE(cusparseDestroy(handle_));
    LOG(debug, "free reduction temp storage");
    CUDA_RUNTIME(cudaFree(dTempStorage_));
    LOG(debug, "free reduction total");
    CUDA_RUNTIME(cudaFree(deviceTotal_));
    LOG(debug, "free C row ptrs");
    CUDA_RUNTIME(cudaFree(csrRowPtrC_));
    LOG(debug, "free C col inds");
    CUDA_RUNTIME(cudaFree(csrColIndC_));
    LOG(debug, "free A values");
    CUDA_RUNTIME(cudaFree(csrValA_));
    LOG(debug, "free C vals");
    CUDA_RUNTIME(cudaFree(csrValC_));
  }
};

} // namespace pangolin