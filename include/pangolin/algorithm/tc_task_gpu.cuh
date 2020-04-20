

#pragma once

#include "count.cuh"
#include "pangolin/cuda_cxx/stream.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/dense/buffer.cuh"
#include "search.cuh"

template <typename TwoColView>
__global__ void task_gpu_count(uint64_t *count, TwoColView adj, typename TwoColView::node_index_type rowStart,
                               typename TwoColView::node_index_type rowStop, typename TwoColView::node_index_type i,
                               typename TwoColView::node_index_type j, typename TwoColView::node_index_type k) {

  typedef typename TwoColView::node_index_type NodeIndex;
  typedef typename TwoColView::edge_index_type EdgeIndex;

  uint64_t threadCount = 0;
  // one block per row/node
  for (NodeIndex src = rowStart + blockIdx.x; src < rowStop; src += gridDim.x) {

    auto srcSlice = adj.row_k(src);
    auto dsts = adj.row_j(src);

    // FIXME: it would be cool to be able to do
    /*
      block_for(dsts.begin(), dsts.end())
     */
    for (size_t dstIdx = threadIdx.x; dstIdx < dsts.size(); dstIdx += blockDim.x) {
      auto dst = dsts[dstIdx];
      auto dstSlice = adj.row_k(dst);
      threadCount +=
          pangolin::serial_sorted_count_linear(srcSlice.begin(), srcSlice.end(), dstSlice.begin(), dstSlice.end());
    }
  }

  atomicAdd(count, threadCount);

}

namespace pangolin {

class TaskGPUTC {
private:
  Buffer<uint64_t> count_;
  StreamRef stream_;

public:
  TaskGPUTC(int dev) : count_(1), stream_(dev) {
    count_[0] = 0;
  }

  TaskGPUTC(int dev, cudaStream_t stream) : count_(1), stream_(stream, dev) {
    count_[0] = 0;
  }
  TaskGPUTC(StreamRef stream) : count_(1), stream_(stream) {
    count_[0] = 0;
  }

  /* Move ctor
  */
  TaskGPUTC(TaskGPUTC &&other) : count_(std::move(other.count_)), stream_(std::move(other.stream_)) {}

  TaskGPUTC() : TaskGPUTC(0) {}

  ~TaskGPUTC() {}

  struct Task {
    size_t i;
    size_t j;
    size_t k;
  };

  template <typename Csr> void count_async(const Csr &adj, const size_t dimBlock = 256) {
    count_async(adj, 0, adj.num_rows(), dimBlock);
  }

  template <typename Csr>
  uint64_t count_sync(const Csr &adj, const size_t rowOffset, const size_t n, const size_t dimBlock = 256) {
    count_async(adj, rowOffset, n, dimBlock);
    sync();
    return count();
  }

  template <typename Csr> uint64_t count_sync(const Csr &adj, const size_t dimBlock = 256) {
    count_async(adj, dimBlock);
    sync();
    return count();
  }

  template <typename TwoColCsr> void count_async(const TwoColCsr &adj, const Task &task) {
    typedef typename TwoColCsr::edge_index_type EdgeIndex;
    typedef typename TwoColCsr::node_index_type NodeIndex;

    NodeIndex rowStart = task.i * adj.partition_size();
    NodeIndex rowStop = min((task.i + 1) * adj.partition_size(), adj.num_rows());

    int dimGrid = adj.partition_size();

    LOG(debug, "rows {}-{}", rowStart, rowStop);

    LOG(debug, "launch task_gpu_count<<<{}, {}, 0, {}>>> dev={}", dimGrid, 256, stream_, stream_.device());
    CUDA_RUNTIME(cudaSetDevice(stream_.device()));
    task_gpu_count<<<dimGrid, 256, 0, cudaStream_t(stream_)>>>(count_.data(), adj, rowStart, rowStop, task.i, task.j, task.k);
  }

  void sync() { stream_.sync(); }
  uint64_t count() const { 
    return count_[0]; }
  int device() const { return stream_.device(); }
  double kernel_time() const { return 0; }
};

} // namespace pangolin