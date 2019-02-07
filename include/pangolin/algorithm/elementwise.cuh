#pragma once

#include "pangolin/namespace.hpp"

PANGOLIN_NAMESPACE_BEGIN


/*! \brief return (1, index) if search_val is in array between left and right, inclusive
return (0, -1) otherwise
*/
template<typename T>
__device__ static ulonglong2 binary_search(const T *const array, 
    size_t left,
    size_t right,
    const T search_val
) {
    while (left <= right) {
        size_t mid = (left + right) / 2;
        T val = array[mid];
        if (val < search_val) {
            left = mid + 1;
        } else if (val > search_val) {
            if (mid == 0) { // prevent rollover when mid = 0 and right becomes unsigned max
                break;
            } else {
                right = mid - 1;
            }
        } else { // val == search_val
            return make_ulonglong2(1, mid);
        }
    }
    return make_ulonglong2(0, (unsigned long long)(-1));
}


/*! non-zero elements in outer product of two sparse vectors
*/
/*
template <IndexType, size_t BLOCK_DIM_X, size_t BLOCK_DIM_Y>
__device__ outer_product_size_block(const IndexType colIndA,
  const IndexType nA,
  const IndexType colIndB,
  const IndexType nB
)
 {

    __shared__ IndexType shIndA[BLOCK_DIM_Y];
    __shared__ ValueType shValA[BLOCK_DIM_Y];
    __shared__ IndexType shIndB[BLOCK_DIM_X];
    __shared__ ValueType shValB[BLOCK_DIM_X];


    for (IndexType i = threadIdx.x; i < nA; i += BLOCK_DIM_X) {
        for (IndexType j = threadIdx.y, j < nB; j += BLOCK_DIM_Y) {
            if (i == 0) {
                shIndB[threadIdx.y] = colIndB[j];
                shValB[threadIdx.y] = colValB[j];
            }
            if (j == 0) {
                shIndA[threadIdx.x] = colIndA[i];
                shValA[threadIdx.x] = colValA[i];
            }
            __syncthreads();

            if (shIndB[threadIdx.y] == shIndA[threadIdx.x]) {
                ValueType partial = shValB[threadIdx.y] * shValA[threadIdx.x];
            }
        }
    }

    if (threadIdx.x = 0) {
        shA[threadIdx.y] = 
    }

    for (size_t i = threadIdx.x; i < nA, i += blockDim.x) {
        bool found = binary_search(colIndB, 0, nB - 1, colIndA[i]);

    }

 }
*/

/*! inner product of sparse A and B in-place in A

may put zeros into A

*/
template <size_t BLOCK_DIM_X, typename IndexType, typename ValueType>
__device__ void inner_product_inplace_block(
    const IndexType *indA,
    ValueType *valA,
    const IndexType nA,
    const IndexType *indB,
    const ValueType *valB,
    const IndexType nB
)
{
    // One thread per element of A
     for (IndexType i = threadIdx.x; i < nA; i += BLOCK_DIM_X) {
        printf("searching for %d between 0 and %d\n", indA[i], nB-1);
        ulonglong2 t = binary_search(indB, 0, nB-1, indA[i]);
        bool found = t.x;
        IndexType loc = t.y;

        if (found) {
            valA[i] *= valB[loc];
        } else {
            valA[i] = 0;
        }
     }
}


/*! \brief CSR elementwise matrix multiplication in-place

A = A .* B

may put zeros into A's rows

*/
template <size_t BLOCK_DIM_X, typename IndexType, typename ValueType>
__global__ void csr_elementwise_inplace( const IndexType *csrRowPtrA,
             const IndexType *csrColIndA,  
             ValueType *csrValA,
             const IndexType *csrRowPtrB,
             const IndexType *csrColIndB,
             const ValueType *csrValB,
             const IndexType numRows //<! number of rows in A and B
           ) {
    // const IndexType nnzA = csrRowPtrA[numRows] - csrRowPtrA[0];
    // const IndexType nnzB = csrRowPtrB[numRows] - csrRowPtrB[0];

  // one threadblock per row
    for (IndexType row = blockIdx.x; row < numRows; row += gridDim.x) {
        printf("row %d\n", row);
        IndexType colStartA = csrRowPtrA[row];
        IndexType colEndA = csrRowPtrA[row + 1];
        IndexType colStartB = csrRowPtrB[row];
        IndexType colEndB = csrRowPtrB[row+1];
        printf("A: %d-%d B: %d-%d\n", colStartA, colEndA, colStartB, colEndB);

        inner_product_inplace_block<BLOCK_DIM_X>(
            &csrColIndA[colStartA],
            &csrValA[colStartA],
            colEndA-colStartA,
            &csrColIndB[colStartB],
            &csrValB[colStartB],
            colEndB - colStartB
        );

    }
}


/*! \brief Compress CSR

if tmp == nullptr, figures out how much tmp storage to allocate

else, compress the CSR


*/
template <typename IndexType, typename ValueType>
__global__ void csr_compress( const IndexType *csrRowPtrA,
             const IndexType *csrColIndA,  
             const ValueType *csrValA,
             const IndexType *csrRowPtrB,
             const IndexType *csrColIndB,
             const ValueType *csrValB,
             const IndexType numRows, //<! number of rows in A and B
             void *tmp
           ) {

    if (nullptr == tmp) {

    } else {

    }
}

PANGOLIN_NAMESPACE_END