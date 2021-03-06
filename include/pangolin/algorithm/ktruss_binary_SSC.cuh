#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"


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
__global__ void InitializeArrays_b(int edgeStart, int numEdges, const CsrCooView mat, BCTYPE *keep_l, BCTYPE *keep_h, 
	bool *affected_l, UT *reversed, BCTYPE *prevKept, UT *srcKP, UT *destKP)
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
		keep_h[i]=val;
		prevKept[i] = val;
		affected_l[i] = false;

		reversed[i] = getEdgeId_b(mat, dn, sn);
		srcKP[i] = i;
		destKP[i] = i;
	}
}


__device__ int AffectOthers_b(UT sp, UT dp, BCTYPE* keep, bool *affected, UT *reversed)
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
__global__ void core_binary_indirect(UT *keepPointer, UT *gnumdeleted, UT *gnumaffected, 
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
							numberAffected += AffectOthers_b(sp, dp, keep, affected, reversed);
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
						numberAffected += AffectOthers_b(sp, dp, keep, affected, reversed);
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
				//atomicAdd(gnumaffected, affectedByBlock);
		}

}


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void core_binary_direct(UT *gnumdeleted, UT *gnumaffected, 
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
								numberAffected += AffectOthers_b(sp, dp, keep, affected, reversed);
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
							numberAffected += AffectOthers_b(sp, dp, keep, affected, reversed);
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


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void core_binary_indirect_3d(UT *keepPointer, UT *gnumdeleted, UT *gnumaffected, 
	const UT k_l, const UT k_h, const size_t edgeStart, const size_t numEdges,
  const CsrCooView mat, BCTYPE *keep_l, BCTYPE *keep_h, bool *affected_l, UT *reversed, bool firstTry, const int uMax)
{
	  // kernel call
	typedef typename CsrCooView::index_type Index;
	size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
	
	
	UT numberDeleted_l = 0;
	UT numberAffected_l = 0;
	UT numberDeleted_h = 0;
	__shared__ bool didAffectAnybody[1];

	bool ft = firstTry; //1
	if(0 == threadIdx.x)
	{
			didAffectAnybody[0] = false;
	}
	__syncthreads();

	int startS=0, startD=0,endS=0,endD=0;
	int prevH=0, prevL=0;
	for(int u=0; u<uMax;u++)
	{
		numberDeleted_l = 0;
		numberDeleted_h = 0;
		
	  for (size_t ii = gx + edgeStart; ii < edgeStart + numEdges; ii += BLOCK_DIM_X * gridDim.x) 
	  {
			size_t i = keepPointer[ii];

			char edgeStatus = keep_h[i]?2:(keep_l[i]?1:0);

		  if (edgeStatus>0 && (affected_l[i] || ft))
		  {
				affected_l[i] = false;
			
				int triCount_l = 0;
				int triCount_h = 0;
			
				UT sp = startS==0?mat.rowPtr_[mat.rowInd_[i]]:startS;
				UT send = mat.rowPtr_[mat.rowInd_[i] + 1];

				UT dp =  startD==0?mat.rowPtr_[mat.colInd_[i]]:startD;
				UT dend = mat.rowPtr_[mat.colInd_[i] + 1];
			
				bool firstHit = true;
				int tcLimit = edgeStatus==2? k_h:k_l;

				while ((edgeStatus==2? triCount_h<k_h : triCount_l<k_l) && sp < send && dp < dend)
				{
					UT sv =  mat.colInd_[sp];
					UT dv =  mat.colInd_[dp];

					if (sv == dv)
					{
						if(keep_h[i])
						{
							if (keep_h[sp] && keep_h[dp]) 
							{
								triCount_h++;
								if (firstHit)
								{
									startS = sp;
									startD = dp;
									firstHit = false;
								}
								endS = sp+1;
								endD = dp+1;
							}
						}
					
						if(triCount_l<k_l-2)
						{
							if (keep_l[sp] && keep_l[dp])
							{
									triCount_l++;
									if (firstHit)
									{
										startS = sp;
										startD = dp;
										firstHit = false;
									}
									endS = sp+1;
									endD = dp+1;
							}
						}
					}

					int yy = sp + ((sv <= dv) ? 1:0);
					dp = dp + ((sv >= dv) ? 1:0);
					sp = yy;
				}
				
				//////////////////////////////////////////////////////////////
				UT ir = reversed[i];
				bool affectEdges = false;
				if(keep_l[i] && triCount_l<k_l-2)
				{
					keep_l[i] = false;
					keep_l[ir] = false;
					affectEdges = triCount_l != prevL;
				}
				if(keep_h[i] && triCount_h<k_h-2)
				{
					keep_h[i] = false;
					keep_h[ir] = false;
					affectEdges |= triCount_h != prevH;
				}

				prevH = triCount_h;
				prevL = triCount_l;
				//////////////////////////////////////////////////////////////
			  if (affectEdges)
			  {
				  UT sp = startS;
				  UT dp = startD;
					while (sp < endS && dp < endD)
					{
						UT sv = mat.colInd_[sp];
						UT dv = mat.colInd_[dp];

						if ((sv == dv))
						{
							int y1 = reversed[sp]; 
							int y2 = reversed[dp];

							if (!affected_l[sp] && (keep_l[sp] || keep_h[sp])) 
							{
								affected_l[sp] = true;
								numberAffected_l++;
							}
							if (!affected_l[dp] && (keep_l[dp]|| keep_h[dp])) 
							{
								affected_l[dp] = true;
								numberAffected_l++;
							}
							if (!affected_l[y1] && (keep_l[y1] || keep_h[y1])) 
							{
								affected_l[y1] = true;
								numberAffected_l++;
							}
							if (!affected_l[y2]  && (keep_l[y2] || keep_h[y2])) 
							{
								affected_l[y2] = true;
								numberAffected_l++;
							}
						}
						int yy = sp + ((sv <= dv) ? 1:0);
						dp = dp + ((sv >= dv) ? 1:0);
						sp = yy;
					}
			  }
		}


		  if(!keep_h[i])
				numberDeleted_h++;
				
			if(!keep_l[i])
			  numberDeleted_l++;
		}
		ft=false;
	}

	//Instead of reduction: hope it works
	if(numberAffected_l>0)
			didAffectAnybody[0] = true;
		__syncthreads();
 		
	if (0 == threadIdx.x) 
	{
		if(didAffectAnybody[0])
			gnumaffected[0] = 1;
	}


 	// Block-wide reduction of threadCount
 	typedef cub::BlockReduce<UT, BLOCK_DIM_X> BlockReduce;
 	__shared__ typename BlockReduce::TempStorage tempStorage;
	 UT deletedByBlock_l = BlockReduce(tempStorage).Sum(numberDeleted_l);
	 UT deletedByBlock_h = BlockReduce(tempStorage).Sum(numberDeleted_h);

 	if (0 == threadIdx.x) 
	  {
				atomicAdd(&(gnumdeleted[0]), deletedByBlock_l);
				atomicAdd(&(gnumdeleted[1]), deletedByBlock_h);
		}
}

template <size_t BLOCK_DIM_X>
__global__ void Store_newbounds(const size_t edgeStart, const size_t numEdges, BCTYPE *keep_s, BCTYPE *keep_d, BCTYPE *prevKept)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			prevKept[i] = keep_s[i];
			keep_d[i] = keep_s[i];
			
		}
}

template <size_t BLOCK_DIM_X>
__global__ void Rewind_newbounds(const size_t edgeStart, const size_t numEdges, BCTYPE *keep_l, BCTYPE *keep_h, BCTYPE *prevKept)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			keep_l[i]=prevKept[i];
			keep_h[i]=prevKept[i]; 
		}
}


template <size_t BLOCK_DIM_X>
__global__ void Store(const size_t edgeStart, const size_t numEdges, BCTYPE *keep, BCTYPE *prevKept)
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
__global__ void Rewind(const size_t edgeStart, const size_t numEdges, BCTYPE *keep, BCTYPE *prevKept)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			keep[i]=prevKept[i];
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


  SingleGPU_Ktruss_Binary(int numEdges, int dev) : dev_(dev) {
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

  SingleGPU_Ktruss_Binary() : SingleGPU_Ktruss_Binary(0,0) {}
  
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
		CUDA_RUNTIME(cudaSetDevice(dev_));
		constexpr int dimBlock = 32; //For edges and nodes

		//For Cooperative calls
		/*cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev_);
		int numSM = deviceProp.multiProcessorCount;
		int maxThPSM = deviceProp.maxThreadsPerMultiProcessor;
		const int maxGridDim = numSM * maxThPSM/dimBlock;*/


		bool firstTry = true;
		BCTYPE *keep_l, *keep_h, *prevKept;
		bool *affected_l;
		UT *reversed, *srcKP, *destKP;

		CUDA_RUNTIME(cudaMallocManaged((void **) &keep_l, numEdges*sizeof(BCTYPE)));
		CUDA_RUNTIME(cudaMallocManaged((void **) &keep_h, numEdges*sizeof(BCTYPE)));
		CUDA_RUNTIME(cudaMallocManaged((void **) &prevKept, numEdges*sizeof(BCTYPE)));

		CUDA_RUNTIME(cudaMallocManaged((void **) &affected_l, numEdges*sizeof(bool)));

		CUDA_RUNTIME(cudaMallocManaged((void **) &reversed, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMallocManaged((void **) &srcKP, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMallocManaged((void **) &destKP, numEdges*sizeof(UT)));
		
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
			
	
		//KTRUSS skeleton
		//Initialize Private Data
		InitializeArrays_b<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, mat, keep_l,keep_h, 
			affected_l, reversed, prevKept, srcKP, destKP);
		*selectedOut = numEdges;
		
		int originalKmin = kmin;
		int originalKmax = kmax;

		UT numDeleted_l = 0;
		UT numDeleted_h = 0;
		UT totalEdges = numEdges;
		float minPercentage = 0.8;
		float percDeleted_l = 0.0;
		float percDeleted_h = 0.0;
		bool cond = kmax - kmin > 1;

		bool findNewBounds = false;
		bool startIndirect = true; //numEdges>6000000; //machine dependent
		if(findNewBounds)
		{
			int k_l = kmin*0.8 + kmax*0.2;
			int k_h = kmin*0.4 + kmax*0.6;

			numDeleted_l = 0;
			numDeleted_h = 0;
			firstTry = true;
			gnumaffected[0] = 0;
			assumpAffected = true;
			cudaDeviceSynchronize();

			dimGridEdges =  (*selectedOut + dimBlock - 1) / dimBlock;
			//Single Run to set new bounds
			while(assumpAffected)
			{
				assumpAffected = false;

				core_binary_indirect_3d<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(destKP,gnumdeleted, 
					gnumaffected, k_l, k_h, edgeOffset, *selectedOut,
					mat, keep_l, keep_h, affected_l, reversed, firstTry, 1); //<Tunable: 4>
				cudaDeviceSynchronize();
				firstTry = false;

				if(gnumaffected[0] > 0)
					assumpAffected = true;

				/*if(numDeleted_l == gnumaffected[0] && numDeleted_h==gnumdeleted[1])
					*assumpAffected = false;
				else *assumpAffected = true;*/

				numDeleted_l = gnumdeleted[0];
				numDeleted_h = gnumdeleted[1];
				gnumdeleted[0]=0;
				gnumdeleted[1]=0;
				gnumaffected[0] = 0;
				cudaDeviceSynchronize();
			}
			percDeleted_l= (numDeleted_l + numEdges - *selectedOut)*1.0/numEdges;
			percDeleted_h= (numDeleted_h + numEdges - *selectedOut)*1.0/numEdges;
			if(percDeleted_l==1.0f && percDeleted_h==1.0f)
			{
				Rewind_newbounds<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep_l, keep_h, prevKept);
				kmax = k_l;

			}
			else if(percDeleted_h == 1.0f)
			{
					Store_newbounds<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep_l, keep_h, prevKept);
					kmin=k_l;
					kmax=k_h;
			}
			else
			{
				Store_newbounds<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep_h, keep_l, prevKept);
				kmin=k_h;
			}
		}
		//////////////////////////////////////////////////////////////////////////////////////
		//KTRUSS Computation
		
		bool rewindFound = false;
		bool forwardFound = false;
		int step = 1;
		while (cond)
		{
			//printf("Kmin=%d, Kmax=%d\n", kmin, kmax);
			/*if(rewindFound || step > 1)
			{
				k =  kmin*minPercentage + kmax*(1-minPercentage);
				minPercentage=0.5;
			}
			else k = (kmin + 20*step) >= kmax? kmax-1: kmin+20*step;
			step++;*/

			k =  kmin*minPercentage + kmax*(1-minPercentage);
			printf("k=%d", k);

			if(kmin==k || kmax==k)
				minPercentage=0.5;
			/*if((kmax-kmin)*1.0/(originalKmax-originalKmin) < 0.2)
				minPercentage = 0.5;*/

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
					core_binary_indirect<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(destKP,gnumdeleted, 
						gnumaffected, k, edgeOffset, *selectedOut,
						mat, keep_l, affected_l, reversed, firstTry, 1); 
				}
				else
				{
					core_binary_direct<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(gnumdeleted, 
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
			if(percDeleted_l==1.0f)
			{
				Rewind<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep_l, prevKept);
				kmax = k;
				rewindFound = true;

			}
			else
			{
				Store<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep_l, prevKept);
				kmin=k;
				forwardFound = true;
				//startIndirect = true;
			}

			cudaDeviceSynchronize();
			totalEdges = *selectedOut;
			//Simple stream compaction: no phsical data movements
			if(kmax-kmin>1 && startIndirect)
			{
				void     *d_temp_storage = NULL;
				size_t   temp_storage_bytes = 0;
				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, keep_l, destKP, selectedOut, numEdges);
				cudaMalloc(&d_temp_storage, temp_storage_bytes);
				cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, keep_l, destKP, selectedOut, numEdges);
				cudaFree(d_temp_storage);
				cudaDeviceSynchronize();
			}

			cond = kmax - kmin > 1;
		}

		k= k= numDeleted_l==totalEdges? k-1:k;

		//CUDA_RUNTIME(cudaGetLastError());
		
		cudaFree(keep_l);
		cudaFree(keep_h);
		cudaFree(reversed);
		cudaFree(affected_l);
		cudaFree(prevKept);
		cudaFree(srcKP);
		cudaFree(destKP);
  }

	template <typename CsrCoo> UT findKtrussBinary_sync(int kmin, int kmax, const CsrCoo &mat, const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) {
    findKtrussBinary_async(kmin, kmax, mat, numNodes, numEdges, nodeOffset, edgeOffset);
    sync();
    return count();
	}

	void sync() 
	{	
		CUDA_RUNTIME(cudaSetDevice(dev_));
		CUDA_RUNTIME(cudaStreamSynchronize(stream_)); 
	}

	UT count() const { return k; }
  int device() const { return dev_; }
};

} // namespace pangolin
