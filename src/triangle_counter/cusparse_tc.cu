#include <cmath>
#include <cub/cub.cuh>
#include <memory>
#include <nvToolsExt.h>

#include "pangolin/algorithm/elementwise.cuh"
#include "pangolin/algorithm/fill.cuh"
#include "pangolin/cusparse.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/narrow.hpp"
#include "pangolin/reader/edge_list_reader.hpp"
#include "pangolin/triangle_counter/cusparse_tc.hpp"
#include "pangolin/utilities.hpp"

PANGOLIN_BEGIN_NAMESPACE()

CusparseTC::CusparseTC(Config &c) : descrA_(nullptr), descrC_(nullptr) {

  if (c.gpus_.size() == 0) {
    LOG(critical, "CusparseTC requires 1 GPU");
    exit(-1);
  }

  gpu_ = c.gpus_[0];
  if (c.gpus_.size() > 1) {
    LOG(warn, "CusparseTC requires exactly 1 GPU. Selected GPU {}", gpu_);
  }

  SPDLOG_DEBUG(logger::console, "create CUSparse handle");
  CUSPARSE(cusparseCreate(&handle_));

  int version;
  CUSPARSE(cusparseGetVersion(handle_, &version));
  LOG(info, "CUSparse version {}", version);

  CUSPARSE(cusparseCreateMatDescr(&descrA_));
  CUSPARSE(cusparseCreateMatDescr(&descrC_));
}

void CusparseTC::read_data(const std::string &path) {
  nvtxRangePush(__PRETTY_FUNCTION__);
  LOG(info, "reading {}", path);
  auto *reader = pangolin::EdgeListReader::from_file(path);
  auto edgeList = reader->read_all();
  if (edgeList.size() == 0) {
    LOG(warn, "empty edge list");
  }
  SPDLOG_DEBUG(logger::console, "building A");
  A_ = GPUCSR<int>::from_edgelist(edgeList, [](const Edge &e) {
    return e.second >= e.first; // keep src > dst
  });
  nvtxRangePop();
}

void CusparseTC::setup_data() { assert(sizeof(Int) == sizeof(int)); }

size_t CusparseTC::count() {

  const int m = checked_narrow(A_.num_rows());
  const int n = checked_narrow(A_.max_col() + 1);
  const int k = checked_narrow(A_.max_col() + 1);
  SPDLOG_DEBUG(logger::console, "CUSparse product m={} n={} k={}", m, n, k);

  const cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;

  const int nnzA = checked_narrow(A_.nnz());
  const int *csrRowPtrA = A_.deviceRowPtr();
  const int *csrColIndA = A_.deviceColInd();
  assert(nnzA == csrRowPtrA[m] - csrRowPtrA[0]);
  SPDLOG_DEBUG(logger::console, "A has {} nonzeros", nnzA);

  int *csrRowPtrC = nullptr;
  SPDLOG_DEBUG(logger::console, "allocate {} rows for C", m);
  CUDA_RUNTIME(cudaMallocManaged(&csrRowPtrC, sizeof(int) * (m + 1)));

  SPDLOG_DEBUG(logger::console, "compute C nnzs");
  int nnzC;
  int *nnzTotalDevHostPtr = &nnzC;
  CUSPARSE(cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST));
  CUSPARSE(cusparseXcsrgemmNnz(handle_, transA, transA, m, n, k, descrA_, nnzA,
                               csrRowPtrA, csrColIndA, descrA_, nnzA,
                               csrRowPtrA, csrColIndA, descrC_, csrRowPtrC,
                               nnzTotalDevHostPtr));
  if (nullptr != nnzTotalDevHostPtr) {
    SPDLOG_TRACE(logger::console, "get nnzC from nnzTotalDevHostPtr");
    nnzC = *nnzTotalDevHostPtr;
    assert(nnzC == csrRowPtrC[m] - csrRowPtrC[0]);
  } else {
    int baseC;
    nnzC = csrRowPtrC[m];
    baseC = csrRowPtrC[0];
    nnzC -= baseC;
  }
  SPDLOG_DEBUG(logger::console, "C has {} nonzeros", nnzC);

  int *csrColIndC = nullptr;
  float *csrValC = nullptr;
  SPDLOG_DEBUG(logger::console, "allocate {} B for csrColIndC",
               sizeof(int) * nnzC);
  CUDA_RUNTIME(cudaMallocManaged(&csrColIndC, sizeof(int) * nnzC));
  SPDLOG_DEBUG(logger::console, "allocate {} B for csrValC",
               sizeof(float) * nnzC);
  CUDA_RUNTIME(cudaMallocManaged(&csrValC, sizeof(float) * nnzC));

  float *csrValA = nullptr;
  SPDLOG_DEBUG(logger::console, "allocate/fill {} B for A csrValA",
               sizeof(float) * nnzA);
  CUDA_RUNTIME(cudaMallocManaged(&csrValA, sizeof(float) * nnzA));
  pangolin::device_fill(csrValA, nnzA, 1.0f);
  CUDA_RUNTIME(cudaDeviceSynchronize());

  SPDLOG_DEBUG(logger::console, "cusparseScsrgemm");
  CUSPARSE(cusparseScsrgemm(handle_, transA, transA, m, n, k, descrA_, nnzA,
                            csrValA, csrRowPtrA, csrColIndA, descrA_, nnzA,
                            csrValA, csrRowPtrA, csrColIndA, descrC_, csrValC,
                            csrRowPtrC, csrColIndC));

  SPDLOG_DEBUG(logger::console, "hadamard product");
  // c .*= A
  constexpr size_t dimBlockX = 256;
  const size_t dimGridX = (m + dimBlockX - 1) / dimBlockX;

  pangolin::csr_elementwise_inplace<dimBlockX><<<dimGridX, dimBlockX>>>(
      csrRowPtrC, csrColIndC, csrValC, csrRowPtrA, csrColIndA, csrValA, m);
  CUDA_RUNTIME(cudaGetLastError());

  float *deviceTotal;
  CUDA_RUNTIME(cudaMallocManaged(&deviceTotal, sizeof(*deviceTotal)));
  *deviceTotal = 0;

  // Reduce the final non-zeros
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  SPDLOG_DEBUG(logger::console, "compute reduction storage requirements");
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, csrValC,
                         deviceTotal, nnzC);
  SPDLOG_DEBUG(logger::console, "allocate {} B for temporary reduction storage",
               temp_storage_bytes);
  CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  SPDLOG_DEBUG(logger::console, "device reduction");
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, csrValC,
                         deviceTotal, nnzC);
  SPDLOG_TRACE(logger::console, "free temporary reduction storage",
               temp_storage_bytes);
  CUDA_RUNTIME(cudaFree(d_temp_storage));

  uint64_t total = *deviceTotal;
  SPDLOG_DEBUG(logger::console, "total is {}", total);

  CUDA_RUNTIME(cudaFree(deviceTotal));
  return total;
}

CusparseTC::~CusparseTC() {
  SPDLOG_DEBUG(logger::console, "destroy A");
  CUSPARSE(cusparseDestroyMatDescr(descrA_));
  SPDLOG_DEBUG(logger::console, "destroy C");
  CUSPARSE(cusparseDestroyMatDescr(descrC_));
  SPDLOG_DEBUG(logger::console, "destroy handle");
  CUSPARSE(cusparseDestroy(handle_));
}

PANGOLIN_END_NAMESPACE()