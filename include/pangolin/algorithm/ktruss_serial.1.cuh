#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

#define UT uint32_t
#define  BCTYPE char

	/*! Binary search
  
     \tparam arr  			Pointer to the array
     \tparam l 					Left boundary of arr
     \tparam r   				Right boundary of arr
		 \tparam x				 	Value to search for 
  */ 
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

	/*! Obtain the index of an edge using its two nodes
  
     \tparam mat  			Graph view represented in COO+CSR format
     \tparam sn 				Source node
     \tparam dn   			Destination node
  */ 
template <typename CsrCooView>
__device__ UT getEdgeId_b(const CsrCooView mat, UT sn, const UT dn)
{
	UT index = 0;

	UT start = mat.rowPtr_[sn];
	UT end2 = mat.rowPtr_[sn+1];
	index = binarySearch_b(mat.colInd_, start, end2, dn); // pangolin::binary_search(p, length, dn);
	return index;
}

template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays(int edgeStart, int numEdges, const CsrCooView mat, BCTYPE *keep_l,
	bool *affected_l, UT *reversed, UT *srcKP, UT *destKP)
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

		bool val =sl>1 && dl>1; 
		keep_l[i]=val;
		affected_l[i] = false;

		reversed[i] = getEdgeId_b(mat, dn, sn);
		srcKP[i] = i;
		destKP[i] = i;
	}
}


__device__ int AffectOthers(UT sp, UT dp, BCTYPE* keep, bool *affected, UT *reversed)
{
	int numberAffected = 0;
	int y1 = reversed[sp]; 
	int y2 = reversed[dp];

	if (!affected[sp] /*&& keep[sp]*/)
	{
		affected[sp] = true;
		numberAffected++;
	}
	if (!affected[dp] /*&& keep[dp]*/)
	{
		affected[dp] = true;
		numberAffected++;
	}
	if (!affected[y1] /*&& keep[y1]*/)
	{
		affected[y1] = true;
		numberAffected++;
	}
	if (!affected[y2] /*&& keep[y2]*/)
	{
		affected[y2] = true;
		numberAffected++;
	}

	return numberAffected;
}




template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void core_indirect(UT *keepPointer, UT *gnumdeleted, UT *gnumaffected, 
	const UT k, const size_t edgeStart, const size_t numEdges,
  const CsrCooView mat, BCTYPE *keep, bool *affected, UT *reversed, bool firstTry, const int uMax)
{
	  // kernel call
	typedef typename CsrCooView::index_type Index;
	size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	UT numberDeleted = 0;
	UT numberAffected = 0;
	__shared__ bool didAffectAnybody[1];
	bool ft = firstTry; //1
	if(0 == threadIdx.x)
		didAffectAnybody[0] = false;

	__syncthreads();
	int startS=0, startD=0,endS=0,endD=0;
	for(int z=0;z<uMax;z++)
	{
		numberDeleted = 0;
		for (size_t ii = gx + edgeStart; ii < edgeStart + numEdges; ii += BLOCK_DIM_X * gridDim.x) 
		{
			size_t i = keepPointer[ii];

			if (keep[i] && (affected[i] || ft))
			{
				affected[i] = false;
				int edgeCount = 0;
			
				UT sp = startS==0?mat.rowPtr_[mat.rowInd_[i]]:startS;
				UT send = mat.rowPtr_[mat.rowInd_[i] + 1];

				UT dp = startD==0?mat.rowPtr_[mat.colInd_[i]]:startD;
				UT dend = mat.rowPtr_[mat.colInd_[i] + 1];
			
				bool firstHit = true;
				while (edgeCount<k-2 && sp < send && dp < dend)
				{
					UT sv = /*sp <limit? source[sp -  spBase]:*/ mat.colInd_[sp];
					UT dv =  mat.colInd_[dp];

					if (sv == dv)
					{
						if (keep[sp] && keep[dp])
						{
							edgeCount++;
							if (firstHit)
							{
								startS = sp;
								startD = dp;
								firstHit = false;
							}
		
							bool cond = ((dend - dp) < (k-2-edgeCount)) || ((send - sp) < (k-2-edgeCount)); //fact
							if(!cond)
							{
								endS = sp+1;
								endD = dp+1;
							}
							else
							{
								numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
							}

						}
					}
					int yy = sp + ((sv <= dv) ? 1:0);
					dp = dp + ((sv >= dv) ? 1:0);
					sp = yy;
				}
				
				//////////////////////////////////////////////////////////////
				if (edgeCount < (k-2))
				{
					UT ir = reversed[i];
					keep[i] = false;
					keep[ir] = false;
					
					UT sp = startS;
					UT dp = startD;

					while (edgeCount>0 && sp < endS && dp < endD)
					{
						UT sv = /*sp < limit? source[sp -  spBase]:*/ mat.colInd_[sp];
						UT dv = mat.colInd_[dp];

						if ((sv == dv))
						{
							numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
						}
						int yy = sp + ((sv <= dv) ? 1:0);
						dp = dp + ((sv >= dv) ? 1:0);
						sp = yy;
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
	 
	 //UT affectedByBlock = BlockReduce(tempStorage).Sum(numberAffected);

 	if (0 == threadIdx.x) 
	  {
				atomicAdd(gnumdeleted, deletedByBlock);
				//atomicAdd(gnumaffected, affectedByBlock);
		}

}


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void core_direct(UT *gnumdeleted, UT *gnumaffected, 
	const UT k, const size_t edgeStart, const size_t numEdges,
  const CsrCooView mat, BCTYPE *keep, bool *affected, UT *reversed, bool firstTry, const int uMax)
{
	  // kernel call
	typedef typename CsrCooView::index_type Index;
	size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	UT numberDeleted = 0;
	UT numberAffected = 0;
	__shared__ bool didAffectAnybody[1];
	bool ft = firstTry; //1
	if(0 == threadIdx.x)
		didAffectAnybody[0] = false;

	__syncthreads();
	int startS=0, startD=0,endS=0,endD=0;

		numberDeleted = 0;
	  for (size_t i = gx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x) 
	  {
		  if (keep[i] && (affected[i] || ft))
		  {
			  affected[i] = false;
				int edgeCount = 0;
			
				UT sp = startS==0?mat.rowPtr_[mat.rowInd_[i]]:startS;
				UT send = mat.rowPtr_[mat.rowInd_[i] + 1];

				UT dp = startD==0?mat.rowPtr_[mat.colInd_[i]]:startD;
				UT dend = mat.rowPtr_[mat.colInd_[i] + 1];
			
				bool firstHit = true;
				while (edgeCount<k-2 && sp < send && dp < dend)
				{
					UT sv = /*sp <limit? source[sp -  spBase]:*/ mat.colInd_[sp];
					UT dv =  mat.colInd_[dp];

					if (sv == dv)
					{
						if (keep[sp] && keep[dp])
						{
							edgeCount++;
							if (firstHit)
							{
								startS = sp;
								startD = dp;
								firstHit = false;
							}
		
							bool cond = ((dend - dp) < (k-2-edgeCount)) || ((send - sp) < (k-2-edgeCount)); //fact
							if(!cond)
							{
								endS = sp+1;
								endD = dp+1;
							}
							else
							{
								numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
							}

						}
					}
					int yy = sp + ((sv <= dv) ? 1:0);
					dp = dp + ((sv >= dv) ? 1:0);
					sp = yy;
				}
				
				//////////////////////////////////////////////////////////////
			  if (edgeCount < (k-2))
			  {
					UT ir = reversed[i];
				  keep[i] = false;
					keep[ir] = false;
					
				  UT sp = startS;
				  UT dp = startD;

					while (edgeCount>0 && sp < endS && dp < endD)
					{
						UT sv = /*sp < limit? source[sp -  spBase]:*/ mat.colInd_[sp];
						UT dv = mat.colInd_[dp];

						if ((sv == dv))
						{
							numberAffected += AffectOthers(sp, dp, keep, affected, reversed);
						}
						int yy = sp + ((sv <= dv) ? 1:0);
						dp = dp + ((sv >= dv) ? 1:0);
						sp = yy;
					}
			  }
		  }

		  if(!keep[i])
			  numberDeleted++;
		}
		ft=false;
	

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
	 
	 //UT affectedByBlock = BlockReduce(tempStorage).Sum(numberAffected);

 	if (0 == threadIdx.x) 
	{
		atomicAdd(gnumdeleted, deletedByBlock);
	}

}

namespace pangolin {

class SingleGPU_Ktruss {
private:
	int dev_;
	cudaStream_t stream_;
	  UT *selectedOut;
	  
	  UT *gnumdeleted;
	  UT *gnumaffected;
	  bool assumpAffected;
  
	  //Outputs:
	  //Max k of a complete ktruss kernel
	  int k;
  
	  //Percentage of deleted edges for a specific k
	  float percentage_deleted_k;
  
  public:
	  BCTYPE *gKeep, *gPrevKeep;
	  bool *gAffected;
	  UT *gReveresed;

public:
  SingleGPU_Ktruss(int dev) : dev_(dev) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
	CUDA_RUNTIME(cudaMallocManaged(&gnumdeleted, 2*sizeof(*gnumdeleted)));
	CUDA_RUNTIME(cudaMallocManaged(&gnumaffected, sizeof(*gnumaffected)));
	CUDA_RUNTIME(cudaMallocManaged(&selectedOut, sizeof(*selectedOut)));

	//CUDA_RUNTIME(cudaMalloc(&gKeep, numEdges*sizeof(bool)));
	//CUDA_RUNTIME(cudaMalloc(&gPrevKeep, numEdges*sizeof(bool)));
	//CUDA_RUNTIME(cudaMalloc(&gAffected,numEdges*sizeof(bool)));
	//CUDA_RUNTIME(cudaMalloc(&gReveresed,numEdges*sizeof(UT)));

	zero_async<2>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
	zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting

	CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  SingleGPU_Ktruss() : SingleGPU_Ktruss(0) {}
  

	template <typename CsrCoo> 
	void findKtrussIncremental_async(int kmin, int kmax, const CsrCoo &mat, 
		const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) 
  	{

		CUDA_RUNTIME(cudaSetDevice(dev_));
		constexpr int dimBlock = 32; //For edges and nodes

		bool firstTry = true;
		BCTYPE *keep_l;
		bool *affected_l;
		UT *reversed, *srcKP, *destKP;

		CUDA_RUNTIME(cudaMallocManaged((void **) &keep_l, numEdges*sizeof(BCTYPE)));

		CUDA_RUNTIME(cudaMallocManaged((void **) &affected_l, numEdges*sizeof(bool)));

		CUDA_RUNTIME(cudaMallocManaged((void **) &reversed, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMallocManaged((void **) &srcKP, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMallocManaged((void **) &destKP, numEdges*sizeof(UT)));
		
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
			
	
		//KTRUSS skeleton
		//Initialize Private Data
		InitializeArrays<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, mat, keep_l, 
			affected_l, reversed, srcKP, destKP);
		*selectedOut = numEdges;
		

		UT numDeleted_l = 0;
		UT totalEdges = numEdges;
		float minPercentage = 0.8;
		float percDeleted_l = 0.0;
		bool startIndirect = true; 
	
		k=3;
		while(true)
		{

			//printf("k=%d\n", k);
			numDeleted_l = 0;
			firstTry = true;
			gnumaffected[0] = 0;
			assumpAffected = true;
			cudaDeviceSynchronize();

			dimGridEdges =  (*selectedOut + dimBlock - 1) / dimBlock;
			while(assumpAffected)
			{
				assumpAffected = false;

				if(startIndirect)
				{
					core_indirect<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(destKP,gnumdeleted, 
						gnumaffected, k, edgeOffset, *selectedOut,
						mat, keep_l, affected_l, reversed, firstTry, 2); 
				}
				else
				{
					core_direct<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(gnumdeleted, 
						gnumaffected, k, edgeOffset, *selectedOut,
						mat, keep_l, affected_l, reversed, firstTry, 1);
				}

				cudaDeviceSynchronize();
				firstTry = false;

				if(gnumaffected[0] > 0)
					assumpAffected = true;

				numDeleted_l = gnumdeleted[0];
		
				gnumdeleted[0]=0;
				gnumaffected[0] = 0;
				cudaDeviceSynchronize();
			}

			percDeleted_l= (numDeleted_l + numEdges - *selectedOut)*1.0/numEdges;
			if(percDeleted_l >= 1.0)
			{
					break;
			}
			else
			{
				k++;
				void     *d_temp_storage = NULL;
				size_t   temp_storage_bytes = 0;
				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, keep_l, destKP, selectedOut, numEdges);
				cudaMalloc(&d_temp_storage, temp_storage_bytes);
				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, keep_l, destKP, selectedOut, numEdges);
				cudaFree(d_temp_storage);
				cudaDeviceSynchronize();
				//zero_async<1>(gnumdeleted, dev_, stream_);
				assumpAffected = true;
			}
			cudaDeviceSynchronize();
			totalEdges = *selectedOut;

		}

		//printf("MAX k = %d\n", *k);


    //CUDA_RUNTIME(cudaGetLastError());
  }

  template <typename CsrCoo> UT findKtrussIncremental_sync(int kmin, int kmax, const CsrCoo &mat, const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) {
    findKtrussIncremental_async(kmin, kmax, mat, numNodes, numEdges, nodeOffset, edgeOffset);
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  UT count() const { return k-1; }
  int device() const { return dev_; }
};

} // namespace pangolin
