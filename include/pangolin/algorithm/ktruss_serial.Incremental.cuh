#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.cuh"
#include "search.cuh"


//Misc
struct TriResult
{
	uint64_t startS = 0;
	uint64_t startD = 0;
	uint64_t endS = 0;
	uint64_t endD = 0;
	bool largerThanK = false;
	bool largerThan0 = false;
};

__device__ uint64_t binarySearch(const uint64_t *arr, uint64_t l, uint64_t r, uint64_t x)
{
	if (r >= l) {
		uint64_t mid = l + (r - l) / 2;

		// If the element is present at the middle
		// itself
		if (arr[mid] == x)
			return mid;

		// If element is smaller than mid, then
		// it can only be present in left subarray
		if (arr[mid] > x)
			return binarySearch(arr, l, mid - 1, x);

		// Else the element can only be present
		// in right subarray
		return binarySearch(arr, mid + 1, r, x);
	}

	// We reach here when element is not
	// present in array
	return 0;
}

template <typename CsrCooView>
__device__ uint64_t getEdgeId(const CsrCooView mat, uint64_t sn, uint64_t dn)
{
	uint64_t index = 0;

	uint64_t start = mat.rowPtr_[sn];
	uint64_t end2 = mat.rowPtr_[sn+1];
	index = binarySearch(mat.colInd_, start, end2, dn);
	return index;
}


//This function is so stupid, 1 thread does linear search !!
//I will fix this for sure
template <typename CsrCooView>
__device__ TriResult CountTriangleOneEdge(const uint64_t i, const int k, const CsrCooView mat, bool *deleted, bool *nodeEliminated)
{
	TriResult t; //whether we found k triangles?

	//node
	uint64_t sn = mat.rowInd_[i];
	uint64_t dn = mat.colInd_[i];

	if (nodeEliminated[sn] || nodeEliminated[dn])
	{
		t.largerThan0 = false;
		t.largerThanK = false;
		return t;
	}

	uint64_t edgeCount = 0;

	//Search for intersection
	//pointer
	uint64_t sp = mat.rowPtr_[sn];
	uint64_t dp = mat.rowPtr_[dn];

	uint64_t send = mat.rowPtr_[sn + 1];
	uint64_t dend = mat.rowPtr_[dn + 1];

	//length
	uint64_t sl = send - sp; /*source: end node   - start node*/
	uint64_t dl = dend - dp; /*dest: end node   - start node*/
	bool firstHit = true;
	if(sl>1 && dl>1)
	{
		while (sp < send && dp < dend && edgeCount<k)
		{
			if (mat.colInd_[sp] == mat.colInd_[dp])
			{
				if (!deleted[sp] && !deleted[dp])
				{
					edgeCount++;
					if (firstHit)
					{
						t.startS = sp;
						t.startD = dp;
						firstHit = false;
					}

					t.endS = sp+1;
					t.endD = dp+1;
				}
				++sp;
				++dp;
			}
			else if (mat.colInd_[sp] < mat.colInd_[dp]) {
				++sp;
			}
			else {
				++dp;
			}
		}
	}

	t.largerThan0 = edgeCount > 0;
	t.largerThanK = edgeCount >= k ;

	return t;
}

//nvprof
//Get Kt
//Count tri Per edge
//Intersect
//Main kernel



template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays(int edgeStart, int numEdges, const CsrCooView mat, bool *deleted, bool *affected, uint64_t *reversed)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;

		int ptx = tx + bx*BLOCK_DIM_X;

		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			deleted[i]=false;
			affected[i] = false;
			uint64_t sn = mat.rowInd_[i];
			uint64_t dn = mat.colInd_[i];
			reversed[i] = getEdgeId(mat, dn, sn);
		}
}


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void NodalUpdateKernel(uint64_t *k, int nodeStart, int numNodes, const CsrCooView mat, bool *nodeEliminated)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx + nodeStart; i< nodeStart + numNodes; i+= BLOCK_DIM_X * gridDim.x)
		{
				nodeEliminated[i] = false;
				uint64_t sp = mat.rowPtr_[i];
				uint64_t dp = mat.rowPtr_[i + 1];
				uint64_t count = dp - sp;
				if (!nodeEliminated[i] && count < *k - 1)
				{
					nodeEliminated[i] = true;
				}
		}
}


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void core(uint64_t *gnumdeleted, uint64_t *gnumaffected, bool *globalMtd,bool *assumpAffected,
	uint64_t *kk, const size_t edgeStart, const size_t numEdges,
  const CsrCooView mat, bool *deleted, bool *affected, uint64_t *reversed,bool *nodeEliminated, bool *firstTry)
{

	const uint64_t k = *kk;
	  // kernel call
	  typedef typename CsrCooView::index_type Index;
	  size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	  uint64_t threadCount = 0;
	  uint64_t numberDeleted = 0;
	  uint64_t numberAffected = 0;
	  __shared__ bool didAffectAnybody[1];
	  if(threadIdx.x == 0)
		  didAffectAnybody[0] = false;
		  
	  //edges assigned to this thread
	  for (size_t i = gx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x) 
	  {
		  if (!deleted[i] && (affected[i]==true || (*firstTry)==true ))
		  {
			  affected[i] = false;
			  TriResult t = CountTriangleOneEdge(i, k-2, mat, deleted, nodeEliminated);

			 if (!t.largerThanK)
			  {
				  //node
				  const uint64_t sn = mat.rowInd_[i];
				  const uint64_t dn = mat.colInd_[i];

				  //Search for intersection
				  //pointer
				  uint64_t sp = t.startS;//neighborPointer[sn];
				  uint64_t dp = t.startD;//neighborPointer[dn];

				  const uint64_t send = t.endS; //neighborPointer[sn + 1];
				  const uint64_t dend = t.endD; //neighborPointer[dn + 1];

				  //length
				  const uint64_t sl = send - sp; 
				  const uint64_t dl = dend - dp; 

				  if (sl>1 && dl>1 && t.largerThan0)
				  {
					  while (sp < send && dp < dend)
					  {
						  if ((mat.colInd_[sp] == mat.colInd_[dp]))
						  {
							  int y1 = reversed[sp]; //getEdgeId(neighborPointer, dest, dest[sp], sn);
							  int y2 = reversed[dp]; //getEdgeId(neighborPointer, dest, dest[sp], dn);

							  if ( !affected[sp] && !deleted[sp])
							  {
								  affected[sp] = true;
								  numberAffected++;
							  }
							  if ( !affected[dp] && !deleted[dp])
							  {
								  affected[dp] = true;
								  numberAffected++;
							  }
							  if (!affected[y1] && !deleted[y1])
							  {
								  affected[y1] = true;
								  numberAffected++;
							  }
							  if (!affected[y2] && !deleted[y2])
							  {
								  affected[y2] = true;
								  numberAffected++;
							  }
							  ++sp;
							  ++dp;
						  }
						  else if (mat.colInd_[sp] < mat.colInd_[dp]) {
							  ++sp;
						  }
						  else {
							  ++dp;
						  }
					  }
				  }

				  uint64_t y1 = reversed[i]; //getEdgeId(neighborPointer, dest, dn, sn);
				  deleted[i] = true;
				  deleted[y1] = true;
			  }
		  }

		  if(deleted[i])
			  numberDeleted++;
	  }

	  //Instead of reduction: hope it works
	  if(numberAffected>0)
		  didAffectAnybody[0] = true;

			__syncthreads();

	  // Block-wide reduction of threadCount
	  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
	  __shared__ typename BlockReduce::TempStorage tempStorage;
		uint64_t deletedByBlock = BlockReduce(tempStorage).Sum(numberDeleted);
		//uint64_t affectedBlock = BlockReduce(tempStorage).Sum(numberAffected);

	  if (0 == threadIdx.x) 
	  {
			atomicAdd(gnumdeleted, deletedByBlock);
			//atomicAdd(gnumaffected, affectedBlock);
			if(didAffectAnybody[0])
				*gnumaffected = 1;


		}

}


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void ktruss_kernel(uint64_t *count, //!< [inout] the count, caller should zero
                       const CsrCooView mat, const size_t numEdges, const size_t edgeStart)
{
  typedef typename CsrCooView::index_type Index;

  size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
  uint64_t threadCount = 0;

  for (size_t i = gx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x) 
  {
  
    const Index src = mat.rowInd_[i];
    const Index dst = mat.colInd_[i];

    const Index *srcBegin = &mat.colInd_[mat.rowPtr_[src]];
    const Index *srcEnd = &mat.colInd_[mat.rowPtr_[src + 1]];
    const Index *dstBegin = &mat.colInd_[mat.rowPtr_[dst]];
    const Index *dstEnd = &mat.colInd_[mat.rowPtr_[dst + 1]];

    threadCount += pangolin::serial_sorted_count_linear(srcBegin, srcEnd, dstBegin, dstEnd);
  }

  // Block-wide reduction of threadCount
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

  // Add to total count
  if (0 == threadIdx.x) {
    atomicAdd(count, aggregate);
  }
}

namespace pangolin {

class SingleGPU_Ktruss {
private:
  int dev_;
  cudaStream_t stream_;
	uint64_t *count_;


	uint64_t *gnumdeleted;
	uint64_t *gnumaffected;
	
	//globals
	//these two values to be combined
	bool *globalMtd;
	bool *assumpAffected;
	bool *firstTry;

	uint64_t *k;


public:
  SingleGPU_Ktruss(int dev) : dev_(dev), count_(nullptr) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
	CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
	CUDA_RUNTIME(cudaMallocManaged(&globalMtd, sizeof(*globalMtd)));
	CUDA_RUNTIME(cudaMallocManaged(&assumpAffected, sizeof(*assumpAffected)));
	CUDA_RUNTIME(cudaMallocManaged(&firstTry, sizeof(*firstTry)));

	CUDA_RUNTIME(cudaMallocManaged(&gnumdeleted, sizeof(*gnumdeleted)));
	CUDA_RUNTIME(cudaMallocManaged(&gnumaffected, sizeof(*gnumaffected)));

	CUDA_RUNTIME(cudaMallocManaged(&k, sizeof(*k)));

	zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
	zero_async<1>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
	zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  SingleGPU_Ktruss() : SingleGPU_Ktruss(0) {}
  
 
  
  template <typename CsrCoo> void GetMaxK_Incremental(const CsrCoo &mat, bool *nodeEliminated)
  {
  	
  }
  
  
  template <typename CsrCoo> void Store(const CsrCoo &mat, bool *nodeEliminated)
  {
  	
  }
  
  template <typename CsrCoo> void Rewind(const CsrCoo &mat, bool *nodeEliminated)
  {
  	
  }
  
  template <typename CsrCoo> void GetMaxK_Binary(const CsrCoo &mat, bool *nodeEliminated)
  {
  	
  }
  
  
  

	template <typename CsrCoo> 
	void findKtrussIncremental_async(int kmin, int kmax, const CsrCoo &mat, 
		const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) 
  {
		bool *nodeEliminated, *deleted, *affected;
		uint64_t *reversed;

		CUDA_RUNTIME(cudaMalloc((void **) &nodeEliminated, numNodes*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &deleted, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &affected, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMallocManaged((void **) &reversed, numEdges*sizeof(uint64_t)));

    constexpr int dimBlock = 512; //For edges and nodes
		const int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
		const int dimGridNodes = (numNodes + dimBlock - 1) / dimBlock;
    assert(edgeOffset + numEdges <= mat.nnz());
    assert(count_);
    SPDLOG_DEBUG(logger::console, "device = {}, blocks = {}, threads = {}", dev_, dimGridEdges, dimBlock);
	CUDA_RUNTIME(cudaSetDevice(dev_));
	
		//KTRUSS skeleton
		//Initialize Private Data
		InitializeArrays<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, mat, deleted, affected, reversed);
		sync();
		*k=3;
		for(;*k<1000;(*k)++)
		{


			uint64_t numDeleted = 0;
			*firstTry = true;
			//NodalUpdateKernel
			NodalUpdateKernel<dimBlock><<<dimGridNodes,dimBlock,0,stream_>>>(k, nodeOffset, numNodes, mat, nodeEliminated);
			cudaDeviceSynchronize();

			while(*assumpAffected)
			{
				*assumpAffected = false;

				core<dimBlock><<<dimGridNodes,dimBlock,0,stream_>>>(gnumdeleted, gnumaffected,globalMtd,assumpAffected,k, edgeOffset, numEdges,
					mat, deleted, affected, reversed, nodeEliminated, firstTry);
				cudaDeviceSynchronize();

				*firstTry = false;	
				numDeleted = *gnumdeleted;
			
				if(*gnumaffected > 0)
					 *assumpAffected = true;


				//printf("At k = %d, Inside Affected, num deleted=%d, num affected = %d", *k, numDeleted, *gnumaffected);

				zero_async<1>(gnumdeleted, dev_, stream_);
				zero_async<1>(gnumaffected, dev_, stream_);
				cudaDeviceSynchronize();
			}

			if(numDeleted >= numEdges)
			{
					break;
			}
			else
			{
				zero_async<1>(gnumdeleted, dev_, stream_);
				*assumpAffected = true;
			}
			cudaDeviceSynchronize();

			//printf("finished k = %d\n", *k);
		}

		//printf("MAX k = %d\n", *k);


    CUDA_RUNTIME(cudaGetLastError());
  }

  template <typename CsrCoo> uint64_t findKtrussIncremental_sync(int kmin, int kmax, const CsrCoo &mat, const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) {
    findKtrussIncremental_async(kmin, kmax, mat, numNodes, numEdges, nodeOffset, edgeOffset);
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  uint64_t count() const { return *k-1; }
  int device() const { return dev_; }
};

} // namespace pangolin
