#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

#define UT uint32_t


//Misc
struct TriResult
{
	UT startS = 0;
	UT startD = 0;
	UT endS = 0;
	UT endD = 0;
	bool largerThanK = false;
	bool largerThan0 = false;
};

__device__ UT binarySearch(const UT *arr, UT l, UT r, UT x)
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
__device__ UT getEdgeId(const CsrCooView mat, UT sn, UT dn)
{
	UT index = 0;

	UT start = mat.rowPtr_[sn];
	UT end2 = mat.rowPtr_[sn+1];
	index = binarySearch(mat.colInd_, start, end2, dn);
	return index;
}


//This function is so stupid, 1 thread does linear search !!
//I will fix this for sure
template <typename CsrCooView>
__device__ TriResult CountTriangleOneEdge(const UT i, const int k, const CsrCooView mat, bool *keep)
{
	TriResult t; //whether we found k triangles?

	//node
	UT sn = mat.rowInd_[i];
	UT dn = mat.colInd_[i];

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

//nvprof
//Get Kt
//Count tri Per edge
//Intersect
//Main kernel



template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays(int edgeStart, int numEdges, const CsrCooView mat, bool *keep, bool *affected, UT *reversed, UT *src, UT *dest)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;

		int ptx = tx + bx*BLOCK_DIM_X;

		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			keep[i] = true;
			affected[i] = false;
			UT sn = mat.rowInd_[i];
			UT dn = mat.colInd_[i];
			src[i] = i;
			dest[i] = i;
			reversed[i] = getEdgeId(mat, dn, sn);
		}
}


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void core(UT *keepPointer, UT *gnumdeleted, UT *gnumaffected, bool *globalMtd, bool *assumpAffected, const UT k, const size_t edgeStart, 
	const size_t numEdges, const CsrCooView mat, bool *keep, bool *affected, UT *reversed, bool *firstTry)
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
	  for (size_t ii = gx + edgeStart; ii < edgeStart + numEdges; ii += BLOCK_DIM_X * gridDim.x) 
	  {
			size_t i = keepPointer[ii];

		  if (keep[i] && (affected[i]==true || ft==true ))
		  {
			  affected[i] = false;
			  TriResult t = CountTriangleOneEdge(i, k-2, mat, keep);
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
 	/*typedef cub::BlockReduce<UT, BLOCK_DIM_X> BlockReduce;
 	__shared__ typename BlockReduce::TempStorage tempStorage;
 	UT deletedByBlock = BlockReduce(tempStorage).Sum(numberDeleted);

 	if (0 == threadIdx.x) 
	  {
				atomicAdd(gnumdeleted, deletedByBlock);
		}*/

}

namespace pangolin {

class SingleGPU_Ktruss {
private:
  int dev_;
  cudaStream_t stream_;
	UT *count_;
	UT *selectedOut;;


	UT *gnumdeleted;
	UT *gnumaffected;
	
	//globals
	//these two values to be combined
	bool *globalMtd;
	bool *assumpAffected;
	bool *firstTry;
	UT k;

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
	CUDA_RUNTIME(cudaMallocManaged(&selectedOut, sizeof(*selectedOut)));

	//zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
	zero_async<1>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
	zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  SingleGPU_Ktruss() : SingleGPU_Ktruss(0) {}
  

	template <typename CsrCoo> 
	void findKtrussIncremental_async(int kmin, int kmax, const CsrCoo &mat, 
		const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) 
  {



		CUDA_RUNTIME(cudaSetDevice(dev_));

		bool *keep, *affected;
		UT *reversed, *srcKP, *destKP;
		CUDA_RUNTIME(cudaMallocManaged((void **) &keep, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &affected, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc((void **) &reversed, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMalloc((void **) &srcKP, numEdges*sizeof(UT)));
		CUDA_RUNTIME(cudaMalloc((void **) &destKP, numEdges*sizeof(UT)));

    constexpr int dimBlock = 512; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
    //assert(edgeOffset + numEdges <= mat.nnz());
    //assert(count_);
		//SPDLOG_DEBUG(logger::console, "device = {}, blocks = {}, threads = {}", dev_, dimGridEdges, dimBlock);
		

		UT ne=numEdges;
		k=3;
		InitializeArrays<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeOffset, numEdges, mat, keep, affected, reversed, srcKP, destKP);
		*selectedOut = numEdges;
		cudaDeviceSynchronize();

		while(true)
		{
			UT numDeleted = 0;
			*firstTry = true;


			dimGridEdges =  (*selectedOut + dimBlock - 1) / dimBlock;
			//dimGridEdges = dimGridEdges > 768? 768: dimGridEdges;
			//printf("Blocks=%d\n", dimGridEdges);
			//cudaDeviceSynchronize();

			while(*assumpAffected)
			{
				*assumpAffected = false;
				core<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(destKP,gnumdeleted, gnumaffected,globalMtd,assumpAffected,
					k, edgeOffset, *selectedOut,
					mat, keep, affected, reversed, firstTry);

				cudaDeviceSynchronize();

				*firstTry = false;	
				numDeleted = *gnumdeleted;
				if(*gnumaffected > 0)
					 *assumpAffected = true;

				*gnumdeleted=0;
				cudaDeviceSynchronize();
			}
			
			void     *d_temp_storage = NULL;
			size_t   temp_storage_bytes = 0;
			cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, keep, destKP, selectedOut, numEdges);
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, keep, destKP, selectedOut, numEdges);
			cudaFree(d_temp_storage);
		
			//printf("%d, %d, %d\n", *selectedOut, numEdges - numDeleted, numEdges - sum);

			if(*selectedOut == 0)
			{
					break;
			}
			else
			{
				//Attempt simple stream compaction, no physical deletion of edges: using CUB partition
				k++;
				*assumpAffected = true;
			}
			cudaDeviceSynchronize();

			//printf("finished k = %d\n", *k);
		}

	

		//printf("MAX k = %d\n", *k);


		//CUDA_RUNTIME(cudaGetLastError());
		
		 cudaFree(keep);
		 cudaFree(reversed);
		 cudaFree(affected);
		 cudaFree(srcKP);
		 cudaFree(destKP);

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
