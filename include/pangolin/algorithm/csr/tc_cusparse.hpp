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
  int gpu_;

  cusparseHandle_t handle_;
  cusparseMatDescr_t descrA_;
  cusparseMatDescr_t descrC_;

public:
  CUSparseTC(int dev) : gpu_(dev), handle_(nullptr), descrA_(nullptr), descrC_(nullptr) {
    LOG(debug, "create CUSparse handle");
    CUSPARSE(cusparseCreate(&handle_));

    int version;
    CUSPARSE(cusparseGetVersion(handle_, &version));
    LOG(info, "CUSparse version {}", version);

    CUSPARSE(cusparseCreateMatDescr(&descrA_));
    CUSPARSE(cusparseCreateMatDescr(&descrC_));
  }

  CUSparseTC() : CUSparseTC(-1) {}

  template <typename CSR> uint64_t count_sync(const CSR &csr) {
    const int m = csr.num_rows();
    const int n = csr.num_cols();
    const int k = csr.num_cols();
    const int nnz = csr.nnz();
    LOG(debug, "CUSparse product m={} n={} k={}", m, n, k);

    const cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const int nnzA = csr.nnz();
    const int *csrRowPtrA = csr.row_ptr();
    const int *csrColIndA = csr.col_ind();

    assert(nnzA == csrRowPtrA[m] - csrRowPtrA[0]);
    LOG(debug, "A has {} nonzeros", nnzA);

    int *csrRowPtrC = nullptr;
    LOG(debug, "allocate {} rows for C", m);
    CUDA_RUNTIME(cudaMallocManaged(&csrRowPtrC, sizeof(int) * (m + 1)));

    LOG(debug, "compute C nnzs");
    int nnzC;
    int *nnzTotalDevHostPtr = &nnzC;
    CUSPARSE(cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST));
    CUSPARSE(cusparseXcsrgemmNnz(handle_, transA, transA, m, n, k, descrA_, nnzA, csrRowPtrA, csrColIndA, descrA_, nnzA,
                                 csrRowPtrA, csrColIndA, descrC_, csrRowPtrC, nnzTotalDevHostPtr));
    if (nullptr != nnzTotalDevHostPtr) {
      SPDLOG_TRACE(logger::console(), "get nnzC from nnzTotalDevHostPtr");
      nnzC = *nnzTotalDevHostPtr;
      assert(nnzC == csrRowPtrC[m] - csrRowPtrC[0]);
    } else {
      int baseC;
      nnzC = csrRowPtrC[m];
      baseC = csrRowPtrC[0];
      nnzC -= baseC;
    }
    LOG(debug, "C has {} nonzeros", nnzC);

    int *csrColIndC = nullptr;
    float *csrValC = nullptr;
    LOG(debug, "allocate {} B for csrColIndC", sizeof(int) * nnzC);
    CUDA_RUNTIME(cudaMallocManaged(&csrColIndC, sizeof(int) * nnzC));
    LOG(debug, "allocate {} B for csrValC", sizeof(float) * nnzC);
    CUDA_RUNTIME(cudaMallocManaged(&csrValC, sizeof(float) * nnzC));

    float *csrValA = nullptr;
    LOG(debug, "allocate/fill {} B for A csrValA", sizeof(float) * nnzA);
    CUDA_RUNTIME(cudaMallocManaged(&csrValA, sizeof(float) * nnzA));
    pangolin::device_fill(csrValA, nnzA, 1.0f);
    CUDA_RUNTIME(cudaDeviceSynchronize());

    LOG(debug, "cusparseScsrgemm");
    CUSPARSE(cusparseScsrgemm(handle_, transA, transA, m, n, k, descrA_, nnzA, csrValA, csrRowPtrA, csrColIndA, descrA_,
                              nnzA, csrValA, csrRowPtrA, csrColIndA, descrC_, csrValC, csrRowPtrC, csrColIndC));

    LOG(debug, "hadamard product");
    // c .*= A
    constexpr size_t dimBlockX = 256;
    const size_t dimGridX = (m + dimBlockX - 1) / dimBlockX;

    pangolin::csr_elementwise_inplace<dimBlockX>
        <<<dimGridX, dimBlockX>>>(csrRowPtrC, csrColIndC, csrValC, csrRowPtrA, csrColIndA, csrValA, m);
    CUDA_RUNTIME(cudaGetLastError());

    float *deviceTotal;
    CUDA_RUNTIME(cudaMallocManaged(&deviceTotal, sizeof(*deviceTotal)));
    *deviceTotal = 0;

    // Reduce the final non-zeros
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    LOG(debug, "compute reduction storage requirements");
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, csrValC, deviceTotal, nnzC);
    LOG(debug, "allocate {} B for temporary reduction storage", temp_storage_bytes);
    CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    LOG(debug, "device reduction");
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, csrValC, deviceTotal, nnzC);
    SPDLOG_TRACE(logger::console(), "free temporary reduction storage", temp_storage_bytes);
    CUDA_RUNTIME(cudaFree(d_temp_storage));

    uint64_t total = *deviceTotal;
    LOG(debug, "total is {}", total);

    CUDA_RUNTIME(cudaFree(deviceTotal));
    return total;
  }

  ~CUSparseTC() {
    LOG(debug, "destroy A");
    CUSPARSE(cusparseDestroyMatDescr(descrA_));
    LOG(debug, "destroy C");
    CUSPARSE(cusparseDestroyMatDescr(descrC_));
    LOG(debug, "destroy handle");
    CUSPARSE(cusparseDestroy(handle_));
  }
};

} // namespace pangolin