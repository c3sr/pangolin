#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"


__device__ UT binarySearch_b(const UT *arr, UT l, UT r, UT x)
{
	/*//recursion
	if (r >= l) {
		UT mid = l + (r - l) / 2;

		// If the element is present at the middle
		// itself
		if (arr[mid] == x)
			return mid;

		// If element is smaller than mid, then
		// it can only be present in left subarray
		if (arr[mid] > x)
			return binarySearch_b(arr, l, mid - 1, x);

		// Else the element can only be present
		// in right subarray
		return binarySearch_b(arr, mid + 1, r, x);
	}

	// We reach here when element is not
	// present in array
	return 0;*/

	size_t left = l;
  size_t right = r;
  while (left < right) {
    const size_t mid = (left + right) / 2;
    UT val = arr[mid];
    bool pred =  val < x;
    if (pred) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

template <typename CsrCooView>
__device__ UT getEdgeId_b(const CsrCooView mat, UT sn, const UT dn)
{
	UT index = 0;

	UT start = mat.rowPtr_[sn];
	UT end2 = mat.rowPtr_[sn+1];
	const UT length=  end2-start;
	const UT *p = &(mat.colInd_[start]);
	index = binarySearch_b(mat.colInd_, start, end2, dn); // pangolin::binary_search(p, length, dn);
	return index;
}


//This function is so stupid, 1 thread does linear search !!
//I will fix this for sure
template <typename CsrCooView>
__device__ TriResult CountTriangleOneEdge_b(const UT i, const int k, const CsrCooView mat, bool *keep)
{
	TriResult t; //whether we found k triangles?

	//node
	UT sn = mat.rowInd_[i];
	UT dn = mat.colInd_[i];
	UT edgeCount = 0;

	//Search for intersection
	//pointer
	UT sp = mat.rowPtr_[sn];
	UT dp = mat.rowPtr_[dn];

	UT send = mat.rowPtr_[sn + 1];
	UT dend = mat.rowPtr_[dn + 1];

	t.startS = sp;
	t.startD = dp;
	t.endS = send;
	t.endD = dend;

	//length
	UT sl = send - sp; /*source: end node   - start node*/
	UT dl = dend - dp; /*dest: end node   - start node*/

	int maxTri = sl<dl?sl:dl;

	bool firstHit = true;
	if(sl>1 && dl>1)
	{
		while (sp < send && dp < dend && edgeCount<k)
		{
			if (mat.colInd_[sp] == mat.colInd_[dp])
			{
				if (keep[sp] && keep[dp])
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

template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays_b(int edgeStart, int numEdges, const CsrCooView mat, bool *keep, bool *affected, UT *reversed, bool *prevKept/*, UT *keepPointer*/)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;

		int ptx = tx + bx*BLOCK_DIM_X;

		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			keep[i]=true;
			prevKept[i] = true;
			affected[i] = false;
			UT sn = mat.rowInd_[i];
			UT dn = mat.colInd_[i];
			reversed[i] = getEdgeId_b(mat, dn, sn);
			//keepPointer[i] = i;
		}
}


//For Zaid and Vikram K-truss + Cross-Decomposition
template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays_b(int edgeStart, int numEdges, const CsrCooView mat, bool *keep, bool *affected, UT *reversed, bool *prevKept, const bool *keep_initial/*, UT *keepPointer*/)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;

		int ptx = tx + bx*BLOCK_DIM_X;

		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			keep[i]=keep_initial[i];
			prevKept[i] = keep_initial[i];
			affected[i] = false;
			UT sn = mat.rowInd_[i];
			UT dn = mat.colInd_[i];
			reversed[i] = getEdgeId_b(mat, dn, sn);
			//keepPointer[i] = i;
		}
}


template <size_t BLOCK_DIM_X>
__global__ void Store(const size_t edgeStart, const size_t numEdges, bool *keep, bool *prevKept)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			prevKept[i] = keep[i];
			
		}
}

template <size_t BLOCK_DIM_X>
__global__ void Rewind(const size_t edgeStart, const size_t numEdges, bool *keep, bool *prevKept)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			keep[i]=prevKept[i];
		}
}




template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void core_binary(uint64_t *globalCounter, UT *gnumdeleted, UT *gnumaffected, bool *globalMtd,bool *assumpAffected,
	const UT k, const size_t edgeStart, const size_t numEdges,
  const CsrCooView mat, bool *keep, bool *affected, UT *reversed, bool *firstTry)
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
		  if (keep[i] && (affected[i]==true || ft==true ))
		  {
			  affected[i] = false;
			  TriResult t = CountTriangleOneEdge_b(i, k-2, mat, keep);

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

							  if ( !affected[sp] && keep[sp])
							  {
								  affected[sp] = true;
								  numberAffected++;
							  }
							  if ( !affected[dp] && keep[dp])
							  {
								  affected[dp] = true;
								  numberAffected++;
							  }
							  if (!affected[y1] && keep[y1])
							  {
								  affected[y1] = true;
								  numberAffected++;
							  }
							  if (!affected[y2] && keep[y2])
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
				  keep[i] = false;
				  keep[y1] = false;
			  }
		  }

		  if(!keep[i])
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

	zero_async<1>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
	zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  SingleGPU_Ktruss_Binary() : SingleGPU_Ktruss_Binary(0) {}
  

	template <typename CsrCoo> 
	void findKtrussBinary_async(int kmin, int kmax, const CsrCoo &mat, 
		const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) 
  {
		bool *keep, *affected, *prevKept;
		//UT *keepPointer;

		UT *reversed;
		CUDA_RUNTIME(cudaMalloc((void **) &keep, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &affected, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &prevKept, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &reversed, numEdges*sizeof(UT)));
		//CUDA_RUNTIME(cudaMalloc((void **) &keepPointer, numEdges*sizeof(UT)));
		

    constexpr int dimBlock = 512; //For edges and nodes
		const int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
		const int dimGridNodes = (numNodes + dimBlock - 1) / dimBlock;
    //assert(edgeOffset + numEdges <= mat.nnz());
    //assert(count_);
    //SPDLOG_DEBUG(logger::console, "device = {}, blocks = {}, threads = {}", dev_, dimGridEdges, dimBlock);
		CUDA_RUNTIME(cudaSetDevice(dev_));
	
		//KTRUSS skeleton
		//Initialize Private Data
		InitializeArrays_b<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, mat, keep, affected, reversed, prevKept/*, keepPointer*/);
		cudaDeviceSynchronize();
		
		UT numDeleted = 0;
		float minPercentage = 0.8;
		float maxPercentage = 0.2;
		while (kmax - kmin > 1)
		{
		k =  kmin*minPercentage + kmax*maxPercentage;

		minPercentage = 0.5;
		maxPercentage = 0.5;

			numDeleted = 0;
			*firstTry = true;
			*globalCounter=0;
			*assumpAffected = true;
			cudaDeviceSynchronize();

			while(*assumpAffected)
			{
				*assumpAffected = false;

				core_binary<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(globalCounter,gnumdeleted, gnumaffected,globalMtd,assumpAffected,k, edgeOffset, numEdges,
					mat, keep, affected, reversed, firstTry);
				cudaDeviceSynchronize();

				*firstTry = false;	
				numDeleted = *gnumdeleted;
			
				if(*gnumaffected > 0)
					 *assumpAffected = true;


				//printf("At k = %d, Inside Affected, num deleted=%d, num affected = %d\n", *k, numDeleted, *gnumaffected);


				*gnumdeleted=0;

				//zero_async<1>(gnumdeleted, dev_, stream_);
				//zero_async<1>(gnumaffected, dev_, stream_);
				cudaDeviceSynchronize();
			}

			if(numDeleted >= numEdges)
			{
				Rewind<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep, prevKept);
				kmax = k;
			}
			else
			{

				Store<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep, prevKept);
				kmin = k;
			}
			cudaDeviceSynchronize();

		//	printf("finished k = %d\n", k);
		}

		//printf("MAX k = %d\n", *k);

		k=(numDeleted >= numEdges)?k-1:k;

		//CUDA_RUNTIME(cudaGetLastError());
		

		//cudaFree(deleted);
		//cudaFree(reversed);
		//cudaFree(affected);
		//cudaFree(prevDeleted);
		//cudaFree(keepPointer);
  }

	template <typename CsrCoo> UT findKtrussBinary_sync(int kmin, int kmax, const CsrCoo &mat, const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) {
    findKtrussBinary_async(kmin, kmax, mat, numNodes, numEdges, nodeOffset, edgeOffset);
    sync();
    return count();
	}

//For Zaid and Vikram K-truss + Cross-Decomposition
	template <typename CsrCoo> 
	void findKtrussBinary_async(int kmin, int kmax, const CsrCoo &mat, const bool *keep_initial,
		const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) 
  {
		bool *keep, *affected, *prevKept;

		UT *reversed;
		CUDA_RUNTIME(cudaMalloc((void **) &keep, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &affected, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &prevKept, numEdges*sizeof(bool)));
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
		InitializeArrays_b<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, mat, keep, affected, reversed, prevKept, keep_initial);
		cudaDeviceSynchronize();
		UT numDeleted = 0;
		float minPercentage = 0.8;
		float maxPercentage = 0.2;
		while (kmax - kmin > 1)
		{
		k =  kmin*minPercentage + kmax*maxPercentage;

		minPercentage = 0.5;
		maxPercentage = 0.5;

			numDeleted = 0;
			*firstTry = true;
			*globalCounter=0;
			*assumpAffected = true;
			cudaDeviceSynchronize();

			while(*assumpAffected)
			{
				*assumpAffected = false;

				core_binary<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(globalCounter,gnumdeleted, gnumaffected,globalMtd,assumpAffected,k, edgeOffset, numEdges,
					mat, keep, affected, reversed, firstTry);
				cudaDeviceSynchronize();

				*firstTry = false;	
				numDeleted = *gnumdeleted;
			
				if(*gnumaffected > 0)
					 *assumpAffected = true;

				*gnumdeleted=0;;
				cudaDeviceSynchronize();
			}

			if(numDeleted >= numEdges)
			{
				Rewind<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep, prevKept);
				kmax = k;
			}
			else
			{
				Store<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep, prevKept);
				kmin = k;
			}
			cudaDeviceSynchronize();


			
		}
		k=(numDeleted >= numEdges)?k-1:k;
		//CUDA_RUNTIME(cudaGetLastError());
	
		cudaFree(keep);
		cudaFree(reversed);
		cudaFree(affected);
		cudaFree(prevKept);
  }

//For Zaid and Vikram K-truss + Cross-Decomposition
	template <typename CsrCoo> UT findKtrussBinary_sync(int kmin, int kmax, const CsrCoo &mat, const bool *keep_initial, const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) {
    findKtrussBinary_async(kmin, kmax, mat, keep_initial, numNodes, numEdges,nodeOffset, edgeOffset);
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  UT count() const { return k; }
  int device() const { return dev_; }
};

} // namespace pangolin
