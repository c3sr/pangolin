#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"


struct TriResultE
{
	UT startS = 0;
	UT startD = 0;
	UT endS = 0;
	UT endD = 0;
	bool largerThanK = false;
	bool largerThan0 = false;
};


__device__ UT binarySearch_b(const UT *arr, UT l, UT r, UT x)
{
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
__device__ TriResultE CountTriangleOneEdge_b(const UT i, const int k, const CsrCooView mat, bool *keep)
{
	TriResultE t; //whether we found k triangles?

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
	//length
	UT sl = send - sp; /*source: end node   - start node*/
	UT dl = dend - dp; /*dest: end node   - start node*/

	bool firstHit = true;
	//if(sl>k && dl>k)
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
				//++sp;
				//++dp;
			}
			/*else if (mat.colInd_[sp] < mat.colInd_[dp]) {
				++sp;
			}
			else {
				++dp;
			}*/


			int k = sp + ((mat.colInd_[sp] <= mat.colInd_[dp]) ? 1:0);
			dp = dp + ((mat.colInd_[sp] >= mat.colInd_[dp]) ? 1:0);
			sp = k;


		}
	}

	t.largerThan0 = edgeCount > 0;
	t.largerThanK = edgeCount >= k ;

	return t;
}

template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays_b(int edgeStart, int numEdges, const CsrCooView mat, bool *keep, 
	bool *affected, UT *reversed, bool *prevKept, UT *srcKP, UT *destKP)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;

		int ptx = tx + bx*BLOCK_DIM_X;

		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			//node
			UT sn = mat.rowInd_[i];
			UT dn = mat.colInd_[i];
			//length
			UT sl = mat.rowPtr_[sn + 1] - mat.rowPtr_[sn];
			UT dl = mat.rowPtr_[dn + 1] -  mat.rowPtr_[dn];

			keep[i]=sl>1;
			prevKept[i] = sl>1;
			affected[i] = false;
	
			reversed[i] = getEdgeId_b(mat, dn, sn);
			srcKP[i] = i;
			destKP[i] = i;
		}
}

//For Zaid and Vikram K-truss + Cross-Decomposition
template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays_b(int edgeStart, int numEdges, const CsrCooView mat, bool *keep, bool *affected, UT *reversed, bool *prevKept, UT *srcKP, UT *destKP, const bool *keep_initial)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;

		int ptx = tx + bx*BLOCK_DIM_X;

		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			keep[i]=keep_initial[i];

			UT sn = mat.rowInd_[i];
			UT dn = mat.colInd_[i];
			//length
			UT sl = mat.rowPtr_[sn + 1] - mat.rowPtr_[sn];
			UT dl = mat.rowPtr_[dn + 1] -  mat.rowPtr_[dn];

			keep[i]=sl>1;
			prevKept[i] = sl>1;
			affected[i] = false;
	
			reversed[i] = getEdgeId_b(mat, dn, sn);
			srcKP[i] = i;
			destKP[i] = i;
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
__global__ void core_binary(uint64_t *globalCounter, UT *gnumdeleted, UT *gnumaffected, bool *assumpAffected,
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

	for(int u=0; u<2;u++)
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
		  if (keep[i] && (affected[i] || ft))
		  {
			  affected[i] = false;
				TriResultE t = CountTriangleOneEdge_b(i, k-2, mat, keep);
				
			  if (!t.largerThanK)
			  {
					UT ir = reversed[i];
				  keep[i] = false;
					keep[ir] = false;
					
				  UT sp = t.startS;
				  UT dp = t.startD;

					while (sp < t.endS && dp < t.endD)
					{
						if ((mat.colInd_[sp] == mat.colInd_[dp]))
						{
							int y1 = reversed[sp]; 
							int y2 = reversed[dp];

							if (!affected[sp] && keep[sp])
							{
								affected[sp] = true;
								numberAffected++;
							}
							if (!affected[dp] && keep[dp])
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


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void core_binary_indirect(UT *keepPointer, UT *gnumdeleted, UT *gnumaffected, 
	const UT k, const size_t edgeStart, const size_t numEdges,
  const CsrCooView mat, bool *keep, bool *affected, UT *reversed, bool *firstTry, const int uMax)
{
	  // kernel call
	typedef typename CsrCooView::index_type Index;
	size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	UT numberDeleted = 0;
	UT numberAffected = 0;
	__shared__ bool didAffectAnybody[1];
	bool ft = *firstTry; //1
	if(0 == threadIdx.x)
		didAffectAnybody[0] = false;

	__syncthreads();

	for(int u=0; u<uMax;u++)
	{
		numberDeleted = 0;
	  //numberAffected = 0;	
	  //edges assigned to this thread
	  for (size_t ii = gx + edgeStart; ii < edgeStart + numEdges; ii += BLOCK_DIM_X * gridDim.x) 
	  {
			size_t i = keepPointer[ii];
			UT sn = mat.rowInd_[i];
			UT dn = mat.colInd_[i];

		  if (keep[i] && (affected[i] || ft))
		  {
			  affected[i] = false;
				TriResultE t = CountTriangleOneEdge_b(i, k-2, mat, keep);
				
			  if (!t.largerThanK)
			  {
					UT ir = reversed[i];
				  keep[i] = false;
					keep[ir] = false;
					
				  UT sp = t.startS;
				  UT dp = t.startD;

					while (sp < t.endS && dp < t.endD)
					{
						if ((mat.colInd_[sp] == mat.colInd_[dp]))
						{
							int y1 = reversed[sp]; 
							int y2 = reversed[dp];

							if (!affected[sp] && keep[sp])
							{
								affected[sp] = true;
								numberAffected++;
							}
							if (!affected[dp] && keep[dp])
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
		  }

		  if(!keep[i])
			  numberDeleted++;
		}
		ft=false;
	}

	//Instead of reduction: hope it works
	if(numberAffected>0)
			didAffectAnybody[0] = true;

		__syncthreads();
 		
	if (0 == threadIdx.x) 
	{
		if(didAffectAnybody[0])
			*gnumaffected = 1;
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
	UT *selectedOut;


	UT *gnumdeleted;
	UT *gnumaffected;
	
	//globals
	//these two values to be combined
	bool *assumpAffected;
	bool *firstTry;
	int k;

public:
  SingleGPU_Ktruss_Binary(int dev) : dev_(dev) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
		CUDA_RUNTIME(cudaMallocManaged(&assumpAffected, sizeof(*assumpAffected)));
		CUDA_RUNTIME(cudaMallocManaged(&firstTry, sizeof(*firstTry)));

		CUDA_RUNTIME(cudaMallocManaged(&gnumdeleted, sizeof(*gnumdeleted)));
		CUDA_RUNTIME(cudaMallocManaged(&gnumaffected, sizeof(*gnumaffected)));
		CUDA_RUNTIME(cudaMallocManaged(&selectedOut, sizeof(*selectedOut)));

		zero_async<1>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
		zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting

			CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  SingleGPU_Ktruss_Binary() : SingleGPU_Ktruss_Binary(0) {}
  
	/*! Find K-Truss of undirected graph, K traversing is done in binary-fashion
  
     \tparam kmin  			Min k in binary traversing
     \tparam kmax 			Max k in binary traversing 
     \tparam mat   			COO+CSR represnrarion of graph (Readonly)
		 \tparam numNodes 	Number of nodes in graph
		 \tparam numEdges 	Number of edges in graph
		 \tparam nodeOffset Start index on nodes (For multi GPU or CPU+GPU)
		 \tparam nodeOffset Start index on nodes (For multi GPU or CPU+GPU)
  */ 
	template <typename CsrCoo> 
	void findKtrussBinary_async(int kmin, int kmax, const CsrCoo &mat, 
		const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) 
  {
		constexpr int dimBlock = 32; //For edges and nodes
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev_);
		int numSM = deviceProp.multiProcessorCount;
		int maxThPSM = deviceProp.maxThreadsPerMultiProcessor;
		const int maxGridDim = numSM * maxThPSM/dimBlock;

		bool *keep, *affected, *prevKept;

		UT *reversed, *srcKP, *destKP;
		CUDA_RUNTIME(cudaMalloc((void **) &keep, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &affected, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &prevKept, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &reversed, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMalloc((void **) &srcKP, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMalloc((void **) &destKP, numEdges*sizeof(UT)));
		
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
    //assert(edgeOffset + numEdges <= mat.nnz());
    //assert(count_);
    //SPDLOG_DEBUG(logger::console, "device = {}, blocks = {}, threads = {}", dev_, dimGridEdges, dimBlock);
		CUDA_RUNTIME(cudaSetDevice(dev_));
	
		//KTRUSS skeleton
		//Initialize Private Data
		InitializeArrays_b<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, mat, keep, 
			affected, reversed, prevKept, srcKP, destKP);
		*selectedOut = numEdges;
		cudaDeviceSynchronize();
		
		int originalKmin = kmin;
		int originalKmax = kmax;

		UT numDeleted = 0;
		UT totalEdges = numEdges;
		float minPercentage = 0.75;
		float percDeleted = 0.0;
		bool cond = kmax - kmin > 1;
		int count = 0;
		while (cond)
		{
			k =  kmin*minPercentage + kmax*(1-minPercentage);
		
			/*if((kmax-kmin)*1.0/(originalKmax-originalKmin) < 0.2) //<Tunable: 0.2>
				minPercentage = 0.5;*/
			
			if(k==kmin || k==kmax)
				minPercentage = 0.5;

			numDeleted = 0;
			*firstTry = true;
			*gnumaffected = 0;
			*assumpAffected = true;
			cudaDeviceSynchronize();

			dimGridEdges =  (*selectedOut + dimBlock - 1) / dimBlock;

			int numAffectedLoops = 0;
			while(*assumpAffected)
			{
				*assumpAffected = false;

				core_binary_indirect<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(destKP,gnumdeleted, 
					gnumaffected, k, edgeOffset, *selectedOut,
					mat, keep, affected, reversed, firstTry, 4); //<Tunable: 4>
					
				cudaDeviceSynchronize();

				*firstTry = false;	
				numDeleted = *gnumdeleted;
			
				if(*gnumaffected > 0)
					 *assumpAffected = true;

				*gnumdeleted=0;
				*gnumaffected = 0;
				cudaDeviceSynchronize();

				numAffectedLoops++;
			}

			percDeleted= (numDeleted + numEdges - *selectedOut)*1.0/numEdges;

			/*printf("Blocks = %d, k=%d, numAffectedLoops=%d, NumDeleted=%d, Edges=%d, prog_deleted=%d, prog_total=%d, percentage=%f\n", dimGridEdges, k, 
						numAffectedLoops, (numDeleted + numEdges - *selectedOut), numEdges, numDeleted, *selectedOut, percDeleted);*/
			if(percDeleted==1.0)
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
			totalEdges = *selectedOut;
			//Simple stream compaction: no phsical data movements
			if(kmax-kmin>1)
			{
				void     *d_temp_storage = NULL;
				size_t   temp_storage_bytes = 0;
				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, keep, destKP, selectedOut, numEdges);
				cudaMalloc(&d_temp_storage, temp_storage_bytes);
				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, keep, destKP, selectedOut, numEdges);
				cudaFree(d_temp_storage);
				cudaDeviceSynchronize();
			}

			cond = kmax - kmin > 1;
		}

		k= numDeleted==totalEdges? k-1:k; //*selectedOut==0 ? k-1: k;

		//CUDA_RUNTIME(cudaGetLastError());
		
		cudaFree(keep);
		cudaFree(reversed);
		cudaFree(affected);
		cudaFree(prevKept);
		cudaFree(srcKP);
		cudaFree(destKP);
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
		constexpr int dimBlock = 32; //For edges and nodes
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev_);
		int numSM = deviceProp.multiProcessorCount;
		int maxThPSM = deviceProp.maxThreadsPerMultiProcessor;
		const int maxGridDim = numSM * maxThPSM/dimBlock;

		bool *keep, *affected, *prevKept;

		UT *reversed, *srcKP, *destKP;
		CUDA_RUNTIME(cudaMalloc((void **) &keep, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &affected, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &prevKept, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &reversed, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMalloc((void **) &srcKP, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMalloc((void **) &destKP, numEdges*sizeof(UT)));
		
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
    //assert(edgeOffset + numEdges <= mat.nnz());
    //assert(count_);
    //SPDLOG_DEBUG(logger::console, "device = {}, blocks = {}, threads = {}", dev_, dimGridEdges, dimBlock);
		CUDA_RUNTIME(cudaSetDevice(dev_));
	
		//KTRUSS skeleton
		//Initialize Private Data
		InitializeArrays_b<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, mat, keep, 
			affected, reversed, prevKept, srcKP, destKP,keep_initial);
		*selectedOut = numEdges;
		cudaDeviceSynchronize();
		
		int originalKmin = kmin;
		int originalKmax = kmax;

		UT numDeleted = 0;
		UT totalEdges = numEdges;
		float minPercentage = 0.8;
		float percDeleted = 0.0;
		bool cond = kmax - kmin > 1;
		int count = 0;
		while (cond)
		{
			k =  kmin*minPercentage + kmax*(1-minPercentage);

			if((kmax-kmin)*1.0/(originalKmax-originalKmin) < 0.2)
				minPercentage = 0.5;
			 
			numDeleted = 0;
			*firstTry = true;
			*gnumaffected = 0;
			*assumpAffected = true;
			cudaDeviceSynchronize();

			dimGridEdges =  (*selectedOut + dimBlock - 1) / dimBlock;

			while(*assumpAffected)
			{
				*assumpAffected = false;

				core_binary_indirect<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(destKP,gnumdeleted, 
					gnumaffected, k, edgeOffset, *selectedOut,
					mat, keep, affected, reversed, firstTry, 4/*Tunable*/);
					
				cudaDeviceSynchronize();

				*firstTry = false;	
				numDeleted = *gnumdeleted;
			
				if(*gnumaffected > 0)
					 *assumpAffected = true;

				*gnumdeleted=0;
				*gnumaffected = 0;
				cudaDeviceSynchronize();
			}

			percDeleted= (numDeleted + numEdges - *selectedOut)*1.0/numEdges;
			if(percDeleted==1.0)
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
			totalEdges = *selectedOut;

			//Simple stream compaction: no phsical data movements
			if(kmax-kmin>1)
			{
				void     *d_temp_storage = NULL;
				size_t   temp_storage_bytes = 0;
				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, keep, destKP, selectedOut, numEdges);
				cudaMalloc(&d_temp_storage, temp_storage_bytes);
				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, keep, destKP, selectedOut, numEdges);
				cudaFree(d_temp_storage);
				cudaDeviceSynchronize();
			}
			cond = kmax - kmin > 1;
		}

		k= numDeleted==totalEdges? k-1:k;
		//CUDA_RUNTIME(cudaGetLastError());
		
		cudaFree(keep);
		cudaFree(reversed);
		cudaFree(affected);
		cudaFree(prevKept);
		cudaFree(srcKP);
		cudaFree(destKP);
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
