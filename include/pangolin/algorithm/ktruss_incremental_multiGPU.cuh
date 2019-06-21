#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"



template <size_t BLOCK_DIM_X>
__global__ void InitializeWorkSpace(int numEdges, BCTYPE *keep,  bool *affected)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx; i< numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			keep[i] = true;
			affected[i] = false;
		}
}


template <size_t BLOCK_DIM_X>
__global__ void RebuildArrays(int edgeStart, int numEdges, int totalEdges, UT *rowPtr, UT *rowInd)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	__shared__ UT rows[BLOCK_DIM_X+1];

	int ptx = tx + bx*BLOCK_DIM_X;

	for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
	{
		rows[tx] = rowInd[ptx];

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
		else if(i == totalEdges-1)
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


namespace pangolin {

class MultiGPU_Ktruss_Incremental {
private:
  int dev_;
  cudaStream_t stream_;
	
	

public:
	BCTYPE *gKeep;
	bool *gAffected;
	bool assumpAffected;

	UT *gDstKP;
	UT *gnumdeleted;
	UT *gnumaffected;
	UT *hnumdeleted;
	UT *hnumaffected;

	UT *selectedOut;

	//Outputs:
	//Max k of a complete ktruss kernel
	int k;

	//Percentage of deleted edges for a specific k
	float percentage_deleted_k;

	MultiGPU_Ktruss_Incremental(int numEdges, int dev) : dev_(dev) {
		CUDA_RUNTIME(cudaSetDevice(dev_));
		CUDA_RUNTIME(cudaStreamCreate(&stream_));
		
		CUDA_RUNTIME(cudaMallocHost(&hnumdeleted, 2*sizeof(UT)));
		CUDA_RUNTIME(cudaMallocHost(&hnumaffected, 2*sizeof(UT)));

		CUDA_RUNTIME(cudaMalloc(&gnumdeleted, 2*sizeof(*gnumdeleted)));
		CUDA_RUNTIME(cudaMalloc(&gnumaffected, sizeof(*gnumaffected)));

		zero_async<2>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
		zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting
	}

	MultiGPU_Ktruss_Incremental() : MultiGPU_Ktruss_Incremental(0,0) {}

	void CreateWorkspace(int numEdges)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		CUDA_RUNTIME(cudaMallocManaged(&selectedOut, sizeof(*selectedOut)));
		CUDA_RUNTIME(cudaMallocManaged(&gKeep, numEdges*sizeof(BCTYPE)));
		CUDA_RUNTIME(cudaMalloc(&gAffected,numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc(&gDstKP,numEdges*sizeof(UT)));
	}
	void free()
	{
		cudaFreeHost(hnumaffected);
		cudaFreeHost(hnumdeleted);

		CUDA_RUNTIME(cudaSetDevice(dev_));
		cudaFree(gnumdeleted);
		cudaFree(gnumaffected);
		cudaFree(selectedOut);
		cudaFree(gKeep);
		cudaFree(gAffected);
		cudaFree(gDstKP);
		//cudaFree(gReveresed);
	
	}

	void InitializeWorkSpace_async(int numEdges)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		
		constexpr int dimBlock = 1024; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

		InitializeWorkSpace<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(numEdges, gKeep, gAffected);
	}

	//All GPUs collaborate to initialize this
	void Inialize_Unified_async(int edgeStart, int numEdges, UT *rowPtr, UT *rowInd, UT *colInd, UT *uSrcKp, UT *reversed)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		
		constexpr int dimBlock = 1024; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

		InitializeArrays_Unified<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeStart, numEdges, rowPtr, rowInd, colInd, uSrcKp, reversed);
	}

	void store_async(int numEdges, BCTYPE *baseKeep)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		
		constexpr int dimBlock = 1024; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
		Store_base<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(0, numEdges, gKeep, baseKeep);
	}

	void compact(int numEdges, UT *srcKP)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		void     *d_temp_storage = NULL;
		size_t   temp_storage_bytes = 0;
		cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, gKeep, gDstKP, selectedOut, numEdges, stream_);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, srcKP, gKeep, gDstKP, selectedOut, numEdges, stream_);
		cudaFree(d_temp_storage);
	}




	void setDevice()
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
	}
	void sync() 
	{	
		CUDA_RUNTIME(cudaStreamSynchronize(stream_)); 
	}

	UT count() const { return k; }
	UT numDeleted() const {return hnumdeleted[0]; }
	UT numAffected() const {return hnumaffected[0];}
	float perc_del_k() const { return percentage_deleted_k; }
	int device() const { return dev_; }
	
	cudaStream_t stream() const { return stream_; }
};

} // namespace pangolin
