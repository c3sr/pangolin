#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

//Normal Initialization
template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays_n(int edgeStart, int numEdges, const CsrCooView mat, bool *keep,
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

			bool val =sl>1 && dl>1; 
			keep[i]=val;
			prevKept[i] = val;
			affected[i] = false;
	
		//	reversed[i] = getEdgeId_b(mat, dn, sn);
			srcKP[i] = i;
			destKP[i] = i;
		}
}


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeWorkSpace(const CsrCooView mat, int numEdges, bool *keep, bool *prevKeep, bool *affected, UT *reversed)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx; i< numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			//node
			UT sn = mat.rowInd_[i];
			UT dn = mat.colInd_[i];

			keep[i] = true;
			prevKeep[i] = true;
			affected[i] = false;
			reversed[i] = getEdgeId_b(mat, dn, sn);
		}
}


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeWorkSpace(const CsrCooView mat, int numEdges, bool *keep)
{
		int tx = threadIdx.x;
		int bx = blockIdx.x;
		int ptx = tx + bx*BLOCK_DIM_X;
		for(int i = ptx; i< numEdges; i+= BLOCK_DIM_X * gridDim.x)
		{
			
			keep[i] = true;
		}
}

template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void InitializeArrays_k(int edgeStart, int numEdges, const CsrCooView mat, bool *keep, bool *affected, UT *reversed)
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


namespace pangolin {

class MultiGPU_Ktruss_Binary {
private:
  int dev_;
  cudaStream_t stream_;
	UT *selectedOut;
	

public:
	bool *gKeep, *gPrevKeep;
	bool *gAffected;
	UT *gReveresed;

	UT *gnumdeleted;
	UT *gnumaffected;
	bool assumpAffected;


	UT *hnumdeleted;
	UT *hnumaffected;

	//Outputs:
	//Max k of a complete ktruss kernel
	int k;

	//Percentage of deleted edges for a specific k
	float percentage_deleted_k;


  MultiGPU_Ktruss_Binary(int numEdges, int dev) : dev_(dev) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
		CUDA_RUNTIME(cudaStreamCreate(&stream_));


		/*hnumdeleted =  (UT*)malloc(2*sizeof(UT));
		hnumaffected =  (UT*)malloc(2*sizeof(UT));*/


		CUDA_RUNTIME(cudaMallocHost(&hnumdeleted, 2*sizeof(UT)));
		CUDA_RUNTIME(cudaMallocHost(&hnumdeleted, 2*sizeof(UT)));



		CUDA_RUNTIME(cudaMalloc(&gnumdeleted, 2*sizeof(*gnumdeleted)));
		CUDA_RUNTIME(cudaMalloc(&gnumaffected, sizeof(*gnumaffected)));
		CUDA_RUNTIME(cudaMalloc(&selectedOut, sizeof(*selectedOut)));

		CUDA_RUNTIME(cudaMalloc(&gKeep, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc(&gPrevKeep, numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc(&gAffected,numEdges*sizeof(bool)));
		CUDA_RUNTIME(cudaMalloc(&gReveresed,numEdges*sizeof(UT)));

		zero_async<2>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
		zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting
  }

  MultiGPU_Ktruss_Binary() : MultiGPU_Ktruss_Binary(0,0) {}
  
	template <typename CsrCoo> 
	void findKtrussBinary_k_async(int k, const CsrCoo &mat, 
		const size_t numNodes, const size_t numEdges, const size_t nodeOffset=0, const size_t edgeOffset=0) 
  {

		CUDA_RUNTIME(cudaSetDevice(dev_));

		constexpr int dimBlock = 32; //For edges and nodes
		int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
		UT numDeleted = 0;
		bool firstTry = true;
		
		cudaMemsetAsync(gnumaffected,0,sizeof(UT),stream_);

		assumpAffected = true;
		dimGridEdges =  (numEdges + dimBlock - 1) / dimBlock;
		while(assumpAffected)
		{
			assumpAffected = false;

			core_binary_direct<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(gnumdeleted, 
				gnumaffected, k, edgeOffset, numEdges,
				mat, gKeep, gAffected, gReveresed, firstTry, 2); //<Tunable: 4>

			cudaMemcpyAsync(hnumaffected, gnumaffected, sizeof(UT), cudaMemcpyDeviceToHost, stream_);
			cudaMemcpyAsync(hnumdeleted, gnumdeleted, sizeof(UT), cudaMemcpyDeviceToHost, stream_);
			
			if(hnumaffected[0] > 0)
					assumpAffected = true;

			firstTry = false;
			numDeleted = hnumdeleted[0];
			cudaMemsetAsync(gnumdeleted,0,sizeof(UT),stream_);
			cudaMemsetAsync(gnumaffected,0,sizeof(UT),stream_);
		}

		percentage_deleted_k = (numDeleted)*1.0/numEdges;
  }



		template <typename CsrCoo>
		void InitializeArrays_async(int edgeStart, int numEdges, const CsrCoo &mat, bool *keep, 
			bool *affected, UT *reversed, bool *prevKept, UT *srcKP, UT *destKP)
		{
			CUDA_RUNTIME(cudaSetDevice(dev_));
			constexpr int dimBlock = 32; //For edges and nodes
			int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;
			InitializeArrays_n<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(edgeStart, numEdges, mat, 
					keep, affected,reversed, prevKept, srcKP, destKP);
		}

		template <typename CsrCoo>
		void InitializeWorkSpace_async(const CsrCoo &mat, int numEdges)
		{

			hnumaffected[0] = 1;
			CUDA_RUNTIME(cudaSetDevice(dev_));
			
			constexpr int dimBlock = 1024; //For edges and nodes
			int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

			InitializeWorkSpace<dimBlock><<<dimGridEdges, dimBlock, 0, stream_>>>(mat, numEdges, gKeep, gPrevKeep, gAffected, gReveresed);
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

		template <typename CsrCoo>
		void core_gpu_async(UT *destKP,
			const UT kk,
			size_t edgeOffset,
			const size_t numEdges,
			const CsrCoo &mat, 
			UT *reversed, 
			bool ft, 
			int uMax)
		{

			CUDA_RUNTIME(cudaSetDevice(dev_));
		

			constexpr int dimBlock = 32; //For edges and nodes
			int dimGridEdges = (numEdges + dimBlock - 1) / dimBlock;

			gnumdeleted[0]=0;
			gnumaffected[0] = 0;
			CUDA_RUNTIME(cudaStreamSynchronize(stream_));

			core_binary_direct<dimBlock><<<dimGridEdges,dimBlock,0,stream_>>>(gnumdeleted, 
				gnumaffected, kk, edgeOffset, numEdges,
				mat, gKeep, gAffected, gReveresed, ft, uMax);
		}


	void zero()
	{
		zero_async<2>(gnumdeleted, dev_, stream_); // zero on the device that will do the counting
		zero_async<1>(gnumaffected, dev_, stream_); // zero on the device that will do the counting
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
