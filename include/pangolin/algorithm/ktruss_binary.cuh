#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.cuh"
#include "search.cuh"

#include "pangolin/algorithm/ktruss_serial.Incremental.cuh"

#define UT uint32_t

__device__ UT binarySearch_b(const UT *arr, UT l, UT r, UT x)
{
	if (r >= l) {
		UT mid = l + (r - l) / 2;

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
__device__ UT getEdgeId_b(const CsrCooView mat, UT sn, UT dn)
{
	UT index = 0;

	UT start = mat.rowPtr_[sn];
	UT end2 = mat.rowPtr_[sn+1];
	index = binarySearch_b(mat.colInd_, start, end2, dn);
	return index;
}


//This function is so stupid, 1 thread does linear search !!
//I will fix this for sure
template <typename CsrCooView>
__device__ TriResult CountTriangleOneEdge_b(const UT i, const int k, const CsrCooView mat, bool *deleted)
{
	TriResult t; //whether we found k triangles?

	//node
	UT sn = mat.rowInd_[i];
	UT dn = mat.colInd_[i];

	/*if (nodeEliminated[sn] || nodeEliminated[dn])
	{
		t.largerThan0 = false;
		t.largerThanK = false;
		return t;
	}*/

	UT edgeCount = 0;

	//Search for intersection
	//pointer
	UT sp = mat.rowPtr_[sn];
	uint64_t dp = mat.rowPtr_[dn];

	UT send = mat.rowPtr_[sn + 1];
	UT dend = mat.rowPtr_[dn + 1];

	//length
	UT sl = send - sp; /*source: end node   - start node*/
	UT dl = dend - dp; /*dest: end node   - start node*/
	bool firstHit = true;
	if(sl>k && dl>k)
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
__global__ void InitializeArrays_b(int edgeStart, int numEdges, const CsrCooView mat, bool *deleted, bool *affected, UT *reversed, bool *prevDeleted)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;

		int ptx = tx + bx*BLOCK_DIM_X;

		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			deleted[i]=false;
			prevDeleted[i] = false;
			affected[i] = false;
			UT sn = mat.rowInd_[i];
			UT dn = mat.colInd_[i];
			reversed[i] = getEdgeId_b(mat, dn, sn);
		}
}


template <size_t BLOCK_DIM_X>
__global__ void Store(const size_t edgeStart, const size_t numEdges, bool *deleted, bool *prevDeleted)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			prevDeleted[i] = deleted[i];
			
		}
}

template <size_t BLOCK_DIM_X>
__global__ void Rewind(const size_t edgeStart, const size_t numEdges, bool *deleted, bool *prevDeleted)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			deleted[i]=prevDeleted[i];
		}
}




template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void core_binary(uint64_t *globalCounter, UT *gnumdeleted, UT *gnumaffected, bool *globalMtd,bool *assumpAffected,
	const UT k, const size_t edgeStart, const size_t numEdges,
  const CsrCooView mat, bool *deleted, bool *affected, UT *reversed, bool *firstTry)
{
	  // kernel call
	  typedef typename CsrCooView::index_type Index;
	  size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	  UT numberDeleted = 0;
	  UT numberAffected = 0;
		__shared__ bool didAffectAnybody[1];
		bool ft = *firstTry; //1
		UT na = *gnumaffected;

	for(int u=0; u<6;u++)
	//while(na>0)
	{
		numberDeleted = 0;
	  numberAffected = 0;
		if(gx==0)
			{
				*gnumaffected = 0;
			}

	  if(threadIdx.x == 0)
		{
			didAffectAnybody[0] = false;
		}		
		__syncthreads();	
	  //edges assigned to this thread
	  for (size_t i = gx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x) 
	  {
		  if (!deleted[i] && (affected[i]==true || ft==true ))
		  {
			  affected[i] = false;
			  TriResult t = CountTriangleOneEdge_b(i, k-2, mat, deleted);

			 if (!t.largerThanK)
			  {
				  //node
				  const UT sn = mat.rowInd_[i];
				  const UT dn = mat.colInd_[i];

				  //Search for intersection
				  //pointer
				  UT sp = t.startS;//neighborPointer[sn];
				  UT dp = t.startD;//neighborPointer[dn];

				  const UT send = t.endS; //neighborPointer[sn + 1];
				  const UT dend = t.endD; //neighborPointer[dn + 1];

				  //length
				  const UT sl = send - sp; 
				  const UT dl = dend - dp; 

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

				  UT y1 = reversed[i]; //getEdgeId(neighborPointer, dest, dn, sn);
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
		ft=false;//4

	  if (0 == threadIdx.x) 
	  {
			if(didAffectAnybody[0])
				*gnumaffected = 1;
		}

		//golabaly sycn blocks
		/*if(0 == threadIdx.x)
				atomicAdd(globalCounter, 1);*/
	}


 	// Block-wide reduction of threadCount
 	typedef cub::BlockReduce<UT, BLOCK_DIM_X> BlockReduce;
 	__shared__ typename BlockReduce::TempStorage tempStorage;
 	UT deletedByBlock = BlockReduce(tempStorage).Sum(numberDeleted);

 	if (0 == threadIdx.x) 
	  {
				atomicAdd(gnumdeleted, deletedByBlock);
		}

}


namespace pangolin {

class SingleGPU_Ktruss_Binary {
private:
  int dev_;
  cudaStream_t stream_;
	uint64_t *globalCounter;


	UT *gnumdeleted;
	UT *gnumaffected;
	
	//globals
	//these two values to be combined
	bool *globalMtd;
	bool *assumpAffected;
	bool *firstTry;

	int k;


public:
  SingleGPU_Ktruss_Binary(int dev) : dev_(dev) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
	CUDA_RUNTIME(cudaMallocManaged(&globalMtd, sizeof(*globalMtd)));
	CUDA_RUNTIME(cudaMallocManaged(&assumpAffected, sizeof(*assumpAffected)));
	CUDA_RUNTIME(cudaMallocManaged(&firstTry, sizeof(*firstTry)));

	CUDA_RUNTIME(cudaMallocManaged(&gnumdeleted, sizeof(*gnumdeleted)));
	CUDA_RUNTIME(cudaMallocManaged(&gnumaffected, sizeof(*gnumaffected)));
	CUDA_RUNTIME(cudaMallocManaged(&globalCounter, sizeof(*globalCounter)));

	//zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
	zero_async<1>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
	zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  SingleGPU_Ktruss_Binary() : SingleGPU_Ktruss_Binary(0) {}
  

	template <typename CsrCoo> 
	void findKtrussBinary_async(int kmin, int kmax, const CsrCoo &mat, 
		const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) 
  {


		
		bool *deleted, *affected, *prevDeleted;
		UT *reversed;
		CUDA_RUNTIME(cudaMalloc((void **) &deleted, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &affected, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &prevDeleted, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &reversed, numEdges*sizeof(UT)));
		

    constexpr int dimBlock = 512; //For edges and nodes
		const int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
		const int dimGridNodes = (numNodes + dimBlock - 1) / dimBlock;
    //assert(edgeOffset + numEdges <= mat.nnz());
    //assert(count_);
    //SPDLOG_DEBUG(logger::console, "device = {}, blocks = {}, threads = {}", dev_, dimGridEdges, dimBlock);
		CUDA_RUNTIME(cudaSetDevice(dev_));
	
		//KTRUSS skeleton
		//Initialize Private Data
		InitializeArrays_b<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, mat, deleted, affected, reversed, prevDeleted);
		cudaDeviceSynchronize();
		
		
	float minPercentage = 0.8;
	float maxPercentage = 0.2;
	while (kmax - kmin > 1)
	{
		k =  kmin*minPercentage + kmax*maxPercentage;

		minPercentage = 0.5;
		maxPercentage = 0.5;

			UT numDeleted = 0;
			*firstTry = true;
			*globalCounter=0;
			*assumpAffected = true;
			cudaDeviceSynchronize();

			while(*assumpAffected)
			{
				*assumpAffected = false;

				core_binary<dimBlock><<<dimGridNodes,dimBlock,0,stream_>>>(globalCounter,gnumdeleted, gnumaffected,globalMtd,assumpAffected,k, edgeOffset, numEdges,
					mat, deleted, affected, reversed, firstTry);
				cudaDeviceSynchronize();

				*firstTry = false;	
				numDeleted = *gnumdeleted;
			
				if(*gnumaffected > 0)
					 *assumpAffected = true;


				//printf("At k = %d, Inside Affected, num deleted=%d, num affected = %d\n", *k, numDeleted, *gnumaffected);

				zero_async<1>(gnumdeleted, dev_, stream_);
				//zero_async<1>(gnumaffected, dev_, stream_);
				cudaDeviceSynchronize();
			}

			if(numDeleted >= numEdges)
			{
				Rewind<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, deleted, prevDeleted);
				kmax = k;
			}
			else
			{

				Store<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, deleted, prevDeleted);
				kmin = k;
			}
			cudaDeviceSynchronize();

		//	printf("finished k = %d\n", k);
		}

		//printf("MAX k = %d\n", *k);


    //CUDA_RUNTIME(cudaGetLastError());
  }

  template <typename CsrCoo> UT findKtrussBinary_sync(int kmin, int kmax, const CsrCoo &mat, const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) {
    findKtrussBinary_async(kmin, kmax, mat, numNodes, numEdges, nodeOffset, edgeOffset);
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  UT count() const { return k-1; }
  int device() const { return dev_; }
};

} // namespace pangolin

#undef UT