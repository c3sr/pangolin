#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

//Normal Initialization
template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays_n(int edgeStart, int numEdges, const CsrCooView mat, BCTYPE *keep,
	BCTYPE *affected, UT *reversed, BCTYPE *prevKept, UT *srcKP, UT *destKP)
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
			
			keep[i]=val;
			prevKept[i] = val;
			affected[i] = false;
	
			reversed[i] = getEdgeId_b(mat, dn, sn);

			srcKP[i] = i;
			destKP[i] = i;
		}
}

template <size_t BLOCK_DIM_X>
__global__ void InitializeWorkSpace(int numEdges, BCTYPE *keep, BCTYPE *prevKeep, bool *affected, UT *destKP)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx; i< numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			keep[i] = true;
			prevKeep[i] = true;
			affected[i] = false;
			destKP[i] = i;
		}
}

template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays_k(int edgeStart, int numEdges, const CsrCooView mat, BCTYPE *keep, bool *affected, UT *reversed)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;

		int ptx = tx + bx*BLOCK_DIM_X;

		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{

			//node
			UT sn = mat.rowInd_[i];
			UT dn = mat.colInd_[i];

			keep[i]=true;
			affected[i] = false;
			reversed[i] = getEdgeId_b(mat, dn, sn);
		}
}


template <size_t BLOCK_DIM_X>
__global__ void InitializeArrays_Unified(UT edgeStart, UT numEdges, UT *rowPtr, UT *rowInd, UT *colInd, UT *srcKp, UT *reversed)
{
		UT tx = threadIdx.x;
		UT bx = blockIdx.x;

		UT ptx = tx + bx*BLOCK_DIM_X;
		for(UT i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{

			//node
			UT sn = rowInd[i];
			UT dn = colInd[i];
			srcKp[i] = i;
			reversed[i] = getEdgeId(rowPtr, rowInd, colInd, dn, sn);
		}
}

template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays_Unified(UT edgeStart, UT numEdges, const CsrCooView mat, UT *srcKp, UT *reversed)
{
		UT tx = threadIdx.x;
		UT bx = blockIdx.x;

		UT ptx = tx + bx*BLOCK_DIM_X;
		for(UT i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{

			//node
			UT sn = mat.rowInd_[i];
			UT dn = mat.colInd_[i];
			srcKp[i] = i;
			reversed[i] = getEdgeId_b(mat, dn, sn);
		}
}

template <size_t BLOCK_DIM_X>
__global__ void Store_base(const size_t edgeStart, const size_t numEdges, BCTYPE *keep, BCTYPE *prevKept, BCTYPE *baseKeep)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			prevKept[i] = baseKeep[i];
			keep[i] = baseKeep[i];
		}
}



template <size_t BLOCK_DIM_X>
__global__ void Store_base(const size_t edgeStart, const size_t numEdges, BCTYPE *keep, BCTYPE *baseKeep)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx + edgeStart; i< edgeStart + numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			keep[i] = baseKeep[i];
		}
}



namespace pangolin {

class MultiGPU_Ktruss_Binary {
private:
  int dev_;
  cudaStream_t stream_;
	
	

public:
	BCTYPE *gKeep, *gPrevKeep;
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

	MultiGPU_Ktruss_Binary(int numEdges, int dev) : dev_(dev) {
		CUDA_RUNTIME(cudaSetDevice(dev_));
		CUDA_RUNTIME(cudaStreamCreate(&stream_));
		
		CUDA_RUNTIME(cudaMallocHost(&hnumdeleted, 2*sizeof(UT)));
		CUDA_RUNTIME(cudaMallocHost(&hnumaffected, 2*sizeof(UT)));

		CUDA_RUNTIME(cudaMalloc(&gnumdeleted, 2*sizeof(*gnumdeleted)));
		CUDA_RUNTIME(cudaMalloc(&gnumaffected, sizeof(*gnumaffected)));

		zero_async<2>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
		zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting
	}

	MultiGPU_Ktruss_Binary() : MultiGPU_Ktruss_Binary(0,0) {}

	void CreateWorkspace(int numEdges)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		CUDA_RUNTIME(cudaMallocManaged(&selectedOut, sizeof(*selectedOut)));
		CUDA_RUNTIME(cudaMallocManaged(&gKeep, numEdges*sizeof(BCTYPE)));
		CUDA_RUNTIME(cudaMallocManaged(&gPrevKeep, numEdges*sizeof(BCTYPE)));
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
		cudaFree(gPrevKeep);
		cudaFree(gAffected);
		cudaFree(gDstKP);
		//cudaFree(gReveresed);
	
	}

	
	void InitializeWorkSpace_async(int numEdges)
	{

		hnumaffected[0] = 1;
		CUDA_RUNTIME(cudaSetDevice(dev_));
		
		constexpr int dimBlock = 1024; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

		InitializeWorkSpace<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(numEdges, gKeep, gPrevKeep, gAffected, gDstKP);
	}

	//All GPUs collaborate to initialize this
	template <typename CsrCoo>
	void Inialize_Unified_async(int edgeStart, int numEdges, const CsrCoo &mat, UT *uSrcKp, UT *reversed)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		
		constexpr int dimBlock = 1024; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

		InitializeArrays_Unified<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeStart, numEdges, mat, uSrcKp, reversed);
	}

		//All GPUs collaborate to initialize this
		template <typename CsrCoo>
		void Inialize_Unified_view_async(int edgeStart, int numEdges, const CsrCoo &mat, UT *uSrcKp, UT *reversed)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));
			
			constexpr int dimBlock = 1024; //For edges and nodes
			int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
	
			InitializeArrays_Unified<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeStart, numEdges, mat, uSrcKp, reversed);
		}

	void rewind_async(int numEdges)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		
		constexpr int dimBlock = 1024; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
		Rewind<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(0, numEdges, gKeep, gPrevKeep);
	}

	void store_async(int numEdges)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		
		constexpr int dimBlock = 1024; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

		Store<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(0, numEdges, gKeep, gPrevKeep);
	}

	void store_async(int numEdges, BCTYPE *baseKeep, bool withPrevKeep)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		
		constexpr int dimBlock = 1024; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
		if(withPrevKeep)
			Store_base<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(0, numEdges, gKeep, gPrevKeep, baseKeep);
		else
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



	//For unified memeory
	template <typename CsrCoo>
	void InitializeArrays_u_async(int edgeStart, int numEdges, const CsrCoo &mat, BCTYPE *keep, 
		BCTYPE *affected, UT *reversed, BCTYPE *prevKept, UT *srcKP, UT *destKP)
	{
		CUDA_RUNTIME(cudaSetDevice(dev_));
		constexpr int dimBlock = 1024; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
		InitializeArrays_n<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeStart, numEdges, mat, 
				keep, affected,reversed, prevKept, srcKP, destKP);
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
