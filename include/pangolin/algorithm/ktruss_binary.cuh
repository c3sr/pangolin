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


	/*! Counts Triangles per edge
  
     \tparam i  			Edge index
     \tparam k	 			Number of triangles to search for
		 \tparam mat  		Graph view represented in COO+CSR format
		 \tparam keep			Array to check if an edge is kept or not (deleted)
		 
		 \return TriResulst	has 5 values: whether a # triangles >= k, first and last intersection indices.
	*/ 
	///UNUSED
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
	//UT sl = send - sp; /*source: end node   - start node*/
	//UT dl = dend - dp; /*dest: end node   - start node*/

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
__global__ void InitializeArrays_b(int edgeStart, int numEdges, const CsrCooView mat, bool *keep_l, bool *keep_h, 
	bool *affected_l, UT *reversed, bool *prevKept, UT *srcKP, UT *destKP)
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




template <size_t BLOCK_DIM_X>
__global__ void Store_newbounds(const size_t edgeStart, const size_t numEdges, bool *keep_s, bool *keep_d, bool *prevKept)
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
__global__ void Rewind_newbounds(const size_t edgeStart, const size_t numEdges, bool *keep_l, bool *keep_h, bool *prevKept)
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
__global__ void core_binary_direct(UT *gnumdeleted, UT *gnumaffected, 
	const UT k, const size_t edgeStart, const size_t numEdges,
  const CsrCooView mat, bool *keep, bool *affected, UT *reversed, bool firstTry, const int uMax)
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
	for(int u=0; u<uMax;u++)
	{
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
								//start early
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
__global__ void core_binary_indirect_3d(UT *keepPointer, UT *gnumdeleted, UT *gnumaffected, 
	const UT k_l, const UT k_h, const size_t edgeStart, const size_t numEdges,
  const CsrCooView mat, bool *keep_l, bool *keep_h, bool *affected_l, UT *reversed, bool firstTry, const int uMax)
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


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void core_binary_indirect(UT *keepPointer, UT *gnumdeleted, UT *gnumaffected, 
	const UT k, const size_t edgeStart, const size_t numEdges,
  const CsrCooView mat, bool *keep, bool *affected, UT *reversed, bool firstTry, const int uMax)
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
	for(int u=0; u<uMax;u++)
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
								//start early
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

namespace pangolin {

class SingleGPU_Ktruss_Binary {
private:
  int dev_;
  cudaStream_t stream_;
	UT *selectedOut;
	
	UT *gnumdeleted;
	UT *gnumaffected;
	bool *assumpAffected;

	//Outputs:
	//Max k of a complete ktruss kernel
	int k;

	//Percentage of deleted edges for a specific k
	float percentage_deleted_k;

public:
	bool *gKeep, *gPrevKeep;
	bool *gAffected;
	UT *gReveresed;


  SingleGPU_Ktruss_Binary(int numEdges, int dev) : dev_(dev) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
		CUDA_RUNTIME(cudaMallocManaged(&assumpAffected, 2*sizeof(*assumpAffected)));

		CUDA_RUNTIME(cudaMallocManaged(&gnumdeleted, 2*sizeof(*gnumdeleted)));
		CUDA_RUNTIME(cudaMallocManaged(&gnumaffected, sizeof(*gnumaffected)));
		CUDA_RUNTIME(cudaMallocManaged(&selectedOut, sizeof(*selectedOut)));

		CUDA_RUNTIME(cudaMalloc(&gKeep, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc(&gPrevKeep, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc(&gAffected,numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc(&gReveresed,numEdges*sizeof(UT)));

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
		constexpr int dimBlock = 32; //For edges and nodes

		//For Cooperative calls
		/*cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev_);
		int numSM = deviceProp.multiProcessorCount;
		int maxThPSM = deviceProp.maxThreadsPerMultiProcessor;
		const int maxGridDim = numSM * maxThPSM/dimBlock;*/


		bool firstTry = true;
		bool *keep_l, *keep_h, *prevKept;
		bool *affected_l;
		UT *reversed, *srcKP, *destKP;

		CUDA_RUNTIME(cudaMalloc((void **) &keep_l, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &keep_h, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &prevKept, numEdges*sizeof(bool)));

		CUDA_RUNTIME(cudaMalloc((void **) &affected_l, numEdges*sizeof(bool)));

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
		InitializeArrays_b<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, mat, keep_l,keep_h, 
			affected_l, reversed, prevKept, srcKP, destKP);
		*selectedOut = numEdges;
		cudaDeviceSynchronize();
		
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

		if(findNewBounds)
		{
				int k_l = kmin*0.8 + kmax*0.2;
				int k_h = kmin*0.4 + kmax*0.6;

				numDeleted_l = 0;
				numDeleted_h = 0;
				firstTry = true;
				gnumaffected[0] = 0;
				*assumpAffected = true;
				cudaDeviceSynchronize();

				dimGridEdges =  (*selectedOut + dimBlock - 1) / dimBlock;
				//Single Run to set new bounds
				while(*assumpAffected)
				{
					*assumpAffected = false;

					core_binary_indirect_3d<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(destKP,gnumdeleted, 
						gnumaffected, k_l, k_h, edgeOffset, *selectedOut,
						mat, keep_l, keep_h, affected_l, reversed, firstTry, 1); //<Tunable: 4>
					cudaDeviceSynchronize();
					firstTry = false;

					if(gnumaffected[0] > 0)
							*assumpAffected = true;

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
		while (cond)
		{
			k =  kmin*minPercentage + kmax*(1-minPercentage);

			if((kmax-kmin)*1.0/(originalKmax-originalKmin) < 0.2)
				minPercentage = 0.5;


			/*if(kmin==k || kmax==k)
				minPercentage=0.5;*/

			numDeleted_l = 0;
			firstTry = true;
			gnumaffected[0] = 0;
			*assumpAffected = true;
			cudaDeviceSynchronize();

			dimGridEdges =  (*selectedOut + dimBlock - 1) / dimBlock;

			int numAffectedLoops = 0;
			while(*assumpAffected)
			{
				*assumpAffected = false;

				core_binary_indirect<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(destKP,gnumdeleted, 
					gnumaffected, k, edgeOffset, *selectedOut,
					mat, keep_l, affected_l, reversed, firstTry, 2); //<Tunable: 4>

				cudaDeviceSynchronize();
				firstTry = false;

				if(gnumaffected[0] > 0)
						*assumpAffected = true;

				numDeleted_l = gnumdeleted[0];
		
				gnumdeleted[0]=0;
				gnumaffected[0] = 0;
				cudaDeviceSynchronize();

				numAffectedLoops++;
			}


			percDeleted_l= (numDeleted_l + numEdges - *selectedOut)*1.0/numEdges;

			bool moveOn = false;
			if(percDeleted_l==1.0f)
			{
				Rewind<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep_l, prevKept);
				kmax = k;

			}
			else
			{
				Store<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, keep_l, prevKept);
				kmin=k;
			}

			/*printf("Blocks = %d, k=%d, numAffectedLoops=%d, NumDeleted=%d, Edges=%d, prog_deleted=%d, prog_deleted_h=%d prog_total=%d, percentage=%f,\n", 
					dimGridEdges, k,
						numAffectedLoops, (numDeleted_l + numEdges - *selectedOut),
						numEdges, numDeleted_l, *selectedOut, percDeleted_l);*/

			cudaDeviceSynchronize();
			totalEdges = *selectedOut;
			//Simple stream compaction: no phsical data movements
			if(kmax-kmin>1)
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
