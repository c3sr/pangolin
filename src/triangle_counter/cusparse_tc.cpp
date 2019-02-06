#include <memory>
#include <cmath>
#include <nvToolsExt.h>

#include "pangolin/logger.hpp"
#include "pangolin/triangle_counter/cusparse_tc.hpp"
#include "pangolin/reader/edge_list_reader.hpp"
#include "pangolin/utilities.hpp"
#include "pangolin/cusparse.hpp"
#include "pangolin/narrow.hpp"

PANGOLIN_NAMESPACE_BEGIN

CusparseTC::CusparseTC(Config &c)
{

    if (c.gpus_.size() == 0)
    {
        LOG(critical, "CusparseTC requires 1 GPU");
        exit(-1);
    }
    
    gpu_ = c.gpus_[0];
    if (c.gpus_.size() > 1)
    {
        LOG(warn, "CusparseTC requires exactly 1 GPU. Selected GPU {}", gpu_);
    }

    LOG(debug, "create CUSparse handle");
    CUSPARSE(cusparseCreate(&handle_));

    int version;
    CUSPARSE(cusparseGetVersion(handle_, &version));
    LOG(info, "CUSparse version {}", version);
}

void CusparseTC::read_data(const std::string &path)
{
    nvtxRangePush(__PRETTY_FUNCTION__);
    LOG(info, "reading {}", path);
    auto *reader = pangolin::EdgeListReader::from_file(path);
    auto edgeList = reader->read();
    if (edgeList.size() == 0) {
        LOG(warn, "empty edge list");
    }
    LOG(debug, "building A");
    A_ = GPUCSR<int>::from_edgelist(edgeList);
    LOG(debug, "building B");
    B_ = GPUCSR<int>::from_edgelist(edgeList);
    nvtxRangePop();
}

void CusparseTC::setup_data()
{
    assert(sizeof(Int) == sizeof(int));
}

size_t CusparseTC::count()
{

    const int m = checked_narrow(A_.num_rows());
    const int n = checked_narrow(A_.max_col());
    const int k = checked_narrow(B_.max_col());
    LOG(debug, "CUSparse product m={} n={} k={}", m, n, k);

    const cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;



    cusparseMatDescr_t descrA = nullptr;
    cusparseMatDescr_t descrB = nullptr;
    cusparseMatDescr_t descrC = nullptr;

    CUSPARSE(cusparseCreateMatDescr(&descrA));
    CUSPARSE(cusparseCreateMatDescr(&descrB));
    CUSPARSE(cusparseCreateMatDescr(&descrC));

    const int nnzA = checked_narrow(A_.nnz());
    const int *csrRowPtrA = A_.deviceRowPtr();
    const int *csrColIndA = A_.deviceColInd();
    assert(nnzA == csrRowPtrA[m] - csrRowPtrA[0]);

    const int nnzB = checked_narrow(B_.nnz());
    const int *csrRowPtrB = B_.deviceRowPtr();
    const int *csrColIndB = B_.deviceRowPtr();
    assert(nnzB == csrRowPtrB[m] - csrRowPtrB[0]);

    int *csrRowPtrC = nullptr;
    LOG(debug, "allocate {} rows for C", A_.num_rows());
    CUDA_RUNTIME(cudaMallocManaged(&csrRowPtrC, sizeof(int) * A_.num_rows() + 1));
    
    LOG(debug, "compute C nnzs");
    int nnzC;
    int baseC;
    int *nnzTotalDevHostPtr = &nnzC;
    cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST);
cusparseXcsrgemmNnz(handle_, transA, transB, m, n, k, 
        descrA, nnzA, csrRowPtrA, csrColIndA,
        descrB, nnzB, csrRowPtrB, csrColIndB,
        descrC, csrRowPtrC, nnzTotalDevHostPtr);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    if (nullptr != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;
    } else {
        nnzC = csrRowPtrC[m];
        baseC = csrRowPtrC[0];
        // cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
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
    float *csrValB = nullptr;
    LOG(debug, "allocate/fill {} B for A csrValA", sizeof(float) * nnzA);
    CUDA_RUNTIME(cudaMallocManaged(&csrValA, sizeof(float) * nnzA));
    std::fill(csrValA, csrValA + nnzA, 1.0f);
    LOG(debug, "allocate/fill {} B for B csrValB", sizeof(float) * nnzB);
    CUDA_RUNTIME(cudaMallocManaged(&csrValB, sizeof(float) * nnzB));  
    std::fill(csrValB, csrValB + nnzB, 1.0f);

    CUSPARSE(cusparseScsrgemm(handle_, transA, transB, m, n, k,
        descrA, nnzA,
        csrValA, csrRowPtrA, csrColIndA,
        descrB, nnzB,
        csrValB, csrRowPtrB, csrColIndB,
        descrC,
        csrValC, csrRowPtrC, csrColIndC
    ));
    CUDA_RUNTIME(cudaDeviceSynchronize());


    LOG(debug, "destroy matrix descriptions");
    CUSPARSE(cusparseDestroyMatDescr(descrA));
    CUSPARSE(cusparseDestroyMatDescr(descrB));
    CUSPARSE(cusparseDestroyMatDescr(descrC));

    return 0;
}

CusparseTC::~CusparseTC() {
    CUSPARSE(cusparseDestroy(handle_));
}

PANGOLIN_NAMESPACE_END