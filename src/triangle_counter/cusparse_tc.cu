#include <memory>
#include <cmath>
#include <nvToolsExt.h>
#include <cub/cub.cuh>
#include <cusp/csr_matrix.h>

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
    const int n = checked_narrow(A_.max_col()+1);
    const int k = checked_narrow(B_.max_col()+1);
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
    LOG(debug, "A has {} nonzeros", nnzA);

    const int nnzB = checked_narrow(B_.nnz());
    const int *csrRowPtrB = B_.deviceRowPtr();
    const int *csrColIndB = B_.deviceColInd();
    assert(nnzB == csrRowPtrB[m] - csrRowPtrB[0]);
    LOG(debug, "B has {} nonzeros", nnzB);

    int *csrRowPtrC = nullptr;
    LOG(debug, "allocate {} rows for C", m);
    CUDA_RUNTIME(cudaMallocManaged(&csrRowPtrC, sizeof(int) * (m + 1)));
    
    LOG(debug, "compute C nnzs");
    int nnzC;
    int *nnzTotalDevHostPtr = &nnzC;
    CUSPARSE(cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST));
    CUSPARSE(cusparseXcsrgemmNnz(handle_, transA, transB, m, n, k, 
        descrA, nnzA, csrRowPtrA, csrColIndA,
        descrB, nnzB, csrRowPtrB, csrColIndB,
        descrC, csrRowPtrC, nnzTotalDevHostPtr)
    );
    // CUDA_RUNTIME(cudaDeviceSynchronize());
    if (nullptr != nnzTotalDevHostPtr){
        TRACE("get nnzC from nnzTotalDevHostPtr");
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
    float *csrValB = nullptr;
    LOG(debug, "allocate/fill {} B for A csrValA", sizeof(float) * nnzA);
    CUDA_RUNTIME(cudaMallocManaged(&csrValA, sizeof(float) * nnzA));
    std::fill(csrValA, csrValA + nnzA, 1.0f);
    LOG(debug, "allocate/fill {} B for B csrValB", sizeof(float) * nnzB);
    CUDA_RUNTIME(cudaMallocManaged(&csrValB, sizeof(float) * nnzB));  
    std::fill(csrValB, csrValB + nnzB, 1.0f);

    LOG(debug, "cusparseScsrgemm");
    CUSPARSE(cusparseScsrgemm(handle_, transA, transB, m, n, k,
        descrA, nnzA,
        csrValA, csrRowPtrA, csrColIndA,
        descrB, nnzB,
        csrValB, csrRowPtrB, csrColIndB,
        descrC,
        csrValC, csrRowPtrC, csrColIndC
    ));
    CUDA_RUNTIME(cudaDeviceSynchronize());

    for (size_t i = 0; i < nnzC; ++i) {
        printf("%f ", csrValC[i]);
    }
    printf("\n");


    // use CUSP for element-wise multiplication
    typedef cusp::array1d<int,cusp::device_memory> IndexArray;
    typedef cusp::array1d<float,cusp::device_memory> ValueArray;
    typedef typename IndexArray::view IndexArrayView;
    typedef typename ValueArray::view ValueArrayView;
    cusp::csr_matrix_view<IndexArrayView,IndexArrayView,ValueArrayView> cuspA(m,n,nnzA);
    cusp::csr_matrix_view<IndexArrayView,IndexArrayView,ValueArrayView> cuspB(n,k,nnzB);
    cusp::csr_matrix_view<IndexArrayView,IndexArrayView,ValueArrayView> cuspC(m,k,nnzC);

    float *deviceTotal;
    CUDA_RUNTIME(cudaMallocManaged(&deviceTotal, sizeof(*deviceTotal)));
    *deviceTotal = 0;

    // Determine temporary device storage requirements
    LOG(debug, "device reduction");
    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, csrValC, deviceTotal, nnzC);
    // Allocate temporary storage
    LOG(trace, "allocating {} B for temporary reduction storage", temp_storage_bytes);
    CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, csrValC, deviceTotal, nnzC);
    CUDA_RUNTIME(cudaFree(d_temp_storage));
    
    uint64_t total = *deviceTotal;
    LOG(debug, "total is {}", total);
    
    
    LOG(debug, "destroy matrix descriptions");
    CUSPARSE(cusparseDestroyMatDescr(descrA));
    CUSPARSE(cusparseDestroyMatDescr(descrB));
    CUSPARSE(cusparseDestroyMatDescr(descrC));
    

    CUDA_RUNTIME(cudaFree(deviceTotal));
    return total;
}

CusparseTC::~CusparseTC() {
    CUSPARSE(cusparseDestroy(handle_));
}

PANGOLIN_NAMESPACE_END