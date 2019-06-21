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

__device__ UT getEdgeId(UT *rowPtr, UT *rowInd, UT *colInd, UT sn, const UT dn)
{
	UT index = 0;

	UT start = rowPtr[sn];
	UT end2 = rowPtr[sn+1];
	index = binarySearch_b(colInd, start, end2, dn); // pangolin::binary_search(p, length, dn);
	return index;
}

template <size_t BLOCK_DIM_X>
__global__ void InitializeArrays(int edgeStart, int numEdges, UT *rowPtr, UT *rowInd, UT *colInd,  BCTYPE *keep_l,
	bool *affected_l, UT *reversed, UT *srcKP, UT *destKP)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	int ptx = tx + bx*BLOCK_DIM_X;

	for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
	{
		//node
		UT sn = rowInd[i];
		UT dn = colInd[i];
		//length
		UT sl = rowPtr[sn + 1] - rowPtr[sn];
		UT dl = rowPtr[dn + 1] -  rowPtr[dn];

		bool val =sl>1 && dl>1; 
		keep_l[i]=val;
		affected_l[i] = false;

		reversed[i] = getEdgeId(rowPtr, rowInd, colInd, dn, sn);
		srcKP[i] = i;
		destKP[i] = i;
	}
}


template <size_t BLOCK_DIM_X>
__global__ void RebuildArrays(int edgeStart, int numEdges, UT *rowPtr, UT *rowInd, BCTYPE *keep_l,bool *affected_l)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	__shared__ UT rows[BLOCK_DIM_X+1];

	int ptx = tx + bx*BLOCK_DIM_X;

	for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
	{
		rows[tx] = rowInd[ptx];

		keep_l[i]=true;
		affected_l[i] = false;

		__syncthreads();

		UT end = rows[tx];
		if(i == 0)
		{
			rowPtr[end] = 0;
		}
		else if (tx ==0)
		{
			UT start = rowInd[i-1];
			for(UT j=start+1; j<=end; j++)
			{
				rowPtr[j] = i;
			}
		}
		else if(i == numEdges-1)
		{
			rowPtr[end+1] = i + 1;

			UT start = rows[tx-1];
			for(UT j=start+1; j<=end; j++)
			{
				rowPtr[j] = i;
			}
		}
		else
		{
			UT start = rows[tx-1];
			for(UT j=start+1; j<=end; j++)
			{
				rowPtr[j] = i;
			}

		}
	}

}

template <size_t BLOCK_DIM_X>
__global__ void RebuildReverse(int edgeStart, int numEdges, UT *rowPtr, UT *rowInd, UT *colInd, UT *reversed)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	int ptx = tx + bx*BLOCK_DIM_X;

	for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
	{
		//node
		UT sn = rowInd[i];
		UT dn = colInd[i];
		//length
		UT sl = rowPtr[sn + 1] - rowPtr[sn];
		UT dl = rowPtr[dn + 1] -  rowPtr[dn];
		reversed[i] = getEdgeId(rowPtr, rowInd, colInd, dn, sn);
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

template <size_t BLOCK_DIM_X>
__global__ void core_indirect(UT *keepPointer, UT *gnumdeleted, UT *gnumaffected, 
	const UT k, const size_t edgeStart, const size_t numEdges,
	UT *rowPtr, UT *rowInd, UT *colInd,  BCTYPE *keep, bool *affected, UT *reversed, bool firstTry, const int uMax)
{
	  // kernel call
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

			int srcNode = rowInd[i];
			int dstNode = colInd[i];

		  	if (keep[i] && srcNode<dstNode && (affected[i] || ft))
			{
				affected[i] = false;
				int edgeCount = 0;
			
				UT sp = startS==0?rowPtr[rowInd[i]]:startS;
				UT send = rowPtr[rowInd[i] + 1];

				UT dp = startD==0?rowPtr[colInd[i]]:startD;
				UT dend = rowPtr[colInd[i] + 1];
			
				bool firstHit = true;
				while (edgeCount<k-2 && sp < send && dp < dend)
				{
					UT sv = /*sp <limit? source[sp -  spBase]:*/ colInd[sp];
					UT dv =  colInd[dp];

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
						UT sv = /*sp < limit? source[sp -  spBase]:*/ colInd[sp];
						UT dv = colInd[dp];

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


template <size_t BLOCK_DIM_X>
__global__ void core_direct(UT *gnumdeleted, UT *gnumaffected, 
	const UT k, const size_t edgeStart, const size_t numEdges,
	UT *rowPtr, UT *rowInd, UT *colInd, BCTYPE *keep, bool *affected, UT *reversed, bool firstTry, const int uMax)
{
	  // kernel call
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
			int srcNode = rowInd[i];
			int dstNode = colInd[i];

		  if (keep[i] && srcNode<dstNode && (affected[i] || ft))
		  {
			  affected[i] = false;
				int edgeCount = 0;
			
				UT sp = startS==0?rowPtr[rowInd[i]]:startS;
				UT send = rowPtr[rowInd[i] + 1];

				UT dp = startD==0?rowPtr[colInd[i]]:startD;
				UT dend = rowPtr[colInd[i] + 1];
			
				bool firstHit = true;
				while (edgeCount<k-2 && sp < send && dp < dend)
				{
					UT sv = /*sp <limit? source[sp -  spBase]:*/ colInd[sp];
					UT dv =  colInd[dp];

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
						UT sv = /*sp < limit? source[sp -  spBase]:*/ colInd[sp];
						UT dv = colInd[dp];

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
  
	void findKtrussIncremental_async(int kmin, int kmax, UT *rowPtr, UT *rowInd, UT *colInd, 
		const size_t numNodes, size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) 
  	{

		CUDA_RUNTIME(cudaSetDevice(dev_));
		constexpr int dimBlock = 32; //For edges and nodes

		bool firstTry = true;
		BCTYPE *keep_l;
		bool *affected_l;
		UT *reversed, *srcKP, *destKP;


		UT *ptrSrc, *ptrDst;

		UT *s1, *d1, *s2, *d2;

		CUDA_RUNTIME(cudaMallocManaged((void **) &keep_l, numEdges*sizeof(BCTYPE)));

		CUDA_RUNTIME(cudaMallocManaged((void **) &affected_l, numEdges*sizeof(bool)));

		CUDA_RUNTIME(cudaMallocManaged((void **) &reversed, numEdges*sizeof(UT)));


		CUDA_RUNTIME(cudaMallocManaged((void **) &srcKP, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMallocManaged((void **) &destKP, numEdges*sizeof(UT)));
		
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
			
	
		//KTRUSS skeleton
		//Initialize Private Data
		InitializeArrays<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, rowPtr, rowInd, colInd, keep_l, 
			affected_l, reversed, srcKP, destKP);
		*selectedOut = numEdges;
		

		UT numDeleted_l = 0;
		float minPercentage = 0.8;
		float percDeleted_l = 0.0;
		bool startIndirect = false; 
	
		s1 = rowInd;
		d1 = colInd;

		s2 = srcKP;
		d2 = destKP;

		ptrSrc = s1;
		ptrDst = d1;


		k=3;
		while(true)
		{

			//printf("k=%d\n", k);
			numDeleted_l = 0;
			firstTry = true;
			gnumaffected[0] = 0;
			assumpAffected = true;
			cudaDeviceSynchronize();

			while(assumpAffected)
			{
				assumpAffected = false;

				core_direct<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(gnumdeleted, 
					gnumaffected, k, edgeOffset, *selectedOut,
					rowPtr, ptrSrc, ptrDst,  keep_l, affected_l, reversed, firstTry, 1);
				
				CUDA_RUNTIME(cudaGetLastError());
				cudaDeviceSynchronize();
				firstTry = false;

				if(gnumaffected[0] > 0)
					assumpAffected = true;

				numDeleted_l = gnumdeleted[0];
		
				gnumdeleted[0]=0;
				gnumaffected[0] = 0;
				cudaDeviceSynchronize();
			}

			percDeleted_l= (numDeleted_l)*1.0/(*selectedOut);
			if(percDeleted_l >= 1.0)
			{
					break;
			}
			else
			{
				k++;


			
				void     *d_temp_storage = NULL;
				size_t   temp_storage_bytes = 0;
				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, s1, keep_l, s2, selectedOut, numEdges, stream_);
				CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));

				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, s1, keep_l, s2, selectedOut, numEdges, stream_);
				//CUDA_RUNTIME(cudaFree(d_temp_storage));

				//d_temp_storage = NULL;
				//temp_storage_bytes = 0;
				//cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d1, keep_l, d2, selectedOut, numEdges, stream_);
				//CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d1, keep_l, d2, selectedOut, numEdges, stream_);
				CUDA_RUNTIME(cudaFree(d_temp_storage));

				cudaDeviceSynchronize();
				CUDA_RUNTIME(cudaGetLastError());

				ptrSrc = s2;
				s2 = s1;
				s1 = ptrSrc;

				ptrDst = d2;
				d2 = d1;
				d1 = ptrDst;

				dimGridEdges =  (*selectedOut + dimBlock - 1) / dimBlock;
				numEdges = *selectedOut;
				//printf("Remaining edges = %d\n", *selectedOut);
				//Now let us reinialize
				RebuildArrays<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(0, *selectedOut, rowPtr, ptrSrc, keep_l, affected_l);
				RebuildReverse<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(0, *selectedOut, rowPtr, ptrSrc, ptrDst, reversed);

				cudaDeviceSynchronize();
				CUDA_RUNTIME(cudaGetLastError());
				assumpAffected = true;
			}
			cudaDeviceSynchronize();
		}

		//printf("MAX k = %d\n", *k);


    //CUDA_RUNTIME(cudaGetLastError());
  }

  UT findKtrussIncremental_sync(int kmin, int kmax, UT *rowPtr, UT *rowInd, UT *colInd, const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) {
    findKtrussIncremental_async(kmin, kmax, rowPtr, rowInd, colInd,  numNodes, numEdges, nodeOffset, edgeOffset);
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  UT count() const { return k-1; }
  int device() const { return dev_; }
};

} // namespace pangolin
