#pragma once

#include "pangolin/namespace.hpp"

PANGOLIN_BEGIN_NAMESPACE

// return 1 if search_val is in array between offets left and right, inclusive
template< typename IndexType>
__device__ static bool binary_search(const IndexType *const array, size_t left,
    size_t right, const IndexType search_val) {
    while (left <= right) {
        size_t mid = (left + right) / 2;
        Int val = array[mid];
        if (val < search_val) {
            left = mid + 1;
        } else if (val > search_val) {
            right = mid - 1;
        } else { // val == search_val
            return 1;
        }
    }
    return 0;
}


/*! non-zero elements in outer product of two sparse vectors
*/
template <IndexType, size_t BLOCK_DIM_X, size_t BLOCK_DIM_Y>
__device__ outer_product_size_block( const IndexType colIndA,
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

/*! \brief CSR elementwise matrix multiplication
*/
template <IndexType>
__global__ void elementwise( const IndexType *csrRowPtrA,
             const IndexType *csrColIndA,  
             const ValueType *csrValA,
             const IndexType *csrRowPtrB,
             const IndexType *csrColIndB,
             const ValueType *csrValB,
             const IndexType numRows, //<! number of rows in A and B
             const IndexType *scratch, //<! at least numRows temporary storage required
           ) {
  const IndexType nnzA = csrRowPtrA[numRows] - csrRowPtrA[0];
  const IndexType nnzB = csrRowPtrB[numRows] - csrRowPtrB[0];

  // determine the size of each combined row

  // one threadblock per row
  for (IndexType row = blockIdx.x; row < numRows; row += gridDim.x) {
      IndexType colStartA = csrColIndA[row];
      IndexType colEndA = csrColIndA[row + 1];
    for (IndexType colIndA = colStartA + threadIdx.x; colIndA < colEndA; colIndA += blockDim.x) {
      IndexType colA = csrColIndA[colIndA];
    }
  }




           }

PANGOLIN_END_NAMESPACE