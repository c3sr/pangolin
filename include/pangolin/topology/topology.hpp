#pragma once

#include <climits>
#include <map>
#include <set>
#include <thread>
#include <cstdio>

#include <nvml.h>

#include "pangolin/logger.hpp"
#include "pangolin/numa.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {
namespace topology {

struct CPU;
struct GPU;
struct NUMA;

typedef std::shared_ptr<CPU> CPU_t;
typedef std::shared_ptr<GPU> GPU_t;
typedef std::shared_ptr<NUMA> NUMA_t;

struct GPU {
  int cudaId_;             //!< the CUDA device ID
  unsigned int nvmlIdx_;   //!< the Nvidia management library device index
  std::set<NUMA_t> numas_; //!< the NUMA regions of the CPUs that have affinity with this GPU
  std::set<CPU_t> cpus_;   //!< the CPUs that have affinity with this GPU
  cudaDeviceProp props_;   //!< the CUDA device properties

  GPU(int cudaId, unsigned int nvmlIdx, cudaDeviceProp props) : cudaId_(cudaId), nvmlIdx_(nvmlIdx), props_(props) {}
};

struct CPU {
  int id_;
  NUMA_t numa_;          //!< the NUMA region this CPU is in (null if none or unknown)
  std::set<GPU_t> gpus_; //!< the GPUs with affinity for this CPU

  explicit CPU(int id) : id_(id) {}
};

/*! A NUMA node
 */
struct NUMA {
  int id_;               //!< the id of the numa node (from libnuma)
  std::set<CPU_t> cpus_; //!< the cpus in this numa region
  std::set<GPU_t> gpus_; //!< the GPUs with affinity to at least one CPU in this region

  explicit NUMA(int id) : id_(id) {}
};

namespace detail {

/*! call nvmlInit if it has not been called yet
 */
void init_nvml() {
  static bool init = false;
  if (!init) {
    NVML(nvmlInit());
    init = true;
  }
}

/*! create a GPU_t for each GPU in the system
 */
inline std::vector<GPU_t> make_gpus() {
  std::vector<GPU_t> gpus;

  int ngpus;
  CUDA_RUNTIME(cudaGetDeviceCount(&ngpus));
  for (int cudaId = 0; cudaId < ngpus; ++cudaId) {
    // correlate each GPU's cudaId with nvml index
    char pciBusId[13];
    CUDA_RUNTIME(cudaDeviceGetPCIBusId(pciBusId, 13, cudaId));
    nvmlDevice_t nvmlDevice;
    NVML(nvmlDeviceGetHandleByPciBusId(pciBusId, &nvmlDevice));
    unsigned int nvmlIdx;
    NVML(nvmlDeviceGetIndex(nvmlDevice, &nvmlIdx));

    // get the device properties for each GPU
    cudaDeviceProp props;
    CUDA_RUNTIME(cudaGetDeviceProperties(&props, cudaId));

    gpus.push_back(std::make_shared<GPU>(cudaId, nvmlIdx, props));
  }

  return gpus;
}
} // namespace detail

/*! get the cpus with affinity for gpu

    \return a set<int> of cpus that have affinity with gpu
 */
inline std::set<int> device_cpu_affinity(const int gpu //!< the gpu to get CPU affinity for
) {
  detail::init_nvml();
  nvmlDevice_t nvmlDev = nullptr;
  NVML(nvmlDeviceGetHandleByIndex(gpu, &nvmlDev));

  unsigned int cpuSetSize = static_cast<unsigned int>(
      (std::thread::hardware_concurrency() + sizeof(unsigned long) - 1) / sizeof(unsigned long));
  if (0 == cpuSetSize) {
    LOG(critical, "cpuSetSize is 0, std::thread::hardware_concurrency = {}", std::thread::hardware_concurrency());
    exit(-1);
  }
  std::vector<unsigned long> cpuSet(cpuSetSize);
  NVML(nvmlDeviceGetCpuAffinity(nvmlDev, cpuSetSize, cpuSet.data()));

  std::set<int> cpus;
  constexpr size_t cpusPerWord = sizeof(unsigned long) * CHAR_BIT;
  int cpu = 0;
  for (unsigned long word : cpuSet) {
    for (size_t i = 0; i < cpusPerWord; ++i) {
      if ((word << i) & 0x1) {
        cpus.insert(cpu);
      }
      ++cpu;
    }
  }

  return cpus;
}

/*! return a set of GPUs that have affinity with cpuid
 */
inline std::set<unsigned int> cpu_device_affinity(const unsigned int cpuid) {

  std::set<unsigned int> ret;

  unsigned int count = 0;
  NVML(nvmlSystemGetTopologyGpuSet(cpuid, &count, nullptr));
  std::vector<nvmlDevice_t> devices(count);
  NVML(nvmlSystemGetTopologyGpuSet(cpuid, &count, devices.data()));

  for (nvmlDevice_t device : devices) {
    unsigned int idx;
    NVML(nvmlDeviceGetIndex(device, &idx));
    ret.insert(idx);
  }

  return ret;
}

/*!
 */
inline std::set<int> get_cpus() {
  const size_t ncpus = std::thread::hardware_concurrency();
  std::set<int> ret;
  for (size_t i = 0; i < ncpus; ++i) {
    ret.insert(static_cast<int>(i));
  }
  return ret;
}

/*! get the numa node affinity for the provided set of CPUs
 */
inline std::set<int> cpu_numa_affinity(const std::set<int> &cpus) {
  std::set<int> numas;
  for (const int cpu : cpus) {
    numas.insert(numa::node_of_cpu(cpu));
  }
  return numas;
}

/*! A representation of the CPUs, GPUs, and NUMA regions in the system
 */
struct Topology {
  std::map<int, NUMA_t> numas_;            //!< NUMA regions by OS id
  std::map<int, CPU_t> cpus_;              //!< CPUs by OS id
  std::map<int, GPU_t> cudaGpus_;          //!< GPUs by CUDA device ID
  std::map<unsigned int, GPU_t> nvmlGpus_; //!< gpus by nvml index

  size_t pageSize_; //!< the page size on the system

  /*! return the numa node that the page of ptr is allocated in.
  If NUMA is not supported or otherwise it cannot be determined, return nullptr
  */
  NUMA_t page_numa(void *ptr) {

    assert(pageSize_);

    if (numa::available()) {
      return numas_[numa::node_of_addr(ptr, pageSize_)];
    } else {
      return NUMA_t(nullptr);
    }
  }

  Topology() : pageSize_(0) {}
}; // namespace topology

/*! Lazily build and return the system Topology
 */
Topology &get() {

  // only build topology structure once
  static bool init = false;
  
  static Topology topology;

  if (!init) {
    detail::init_nvml();

    topology.pageSize_ = sysconf(_SC_PAGESIZE);

    // if NUMA is available and installed, add NUMA nodes
    if (numa::available()) {
      for (auto numaId : numa::all_nodes()) {
        auto numa = std::make_shared<NUMA>(numaId);
        topology.numas_.insert(std::make_pair(numaId, numa));
      }
    } else {
      LOG(debug, "NUMA is not available, not adding NUMA nodes to topology");
    }

    const int numCPUs = std::thread::hardware_concurrency();
    // add cpus to topology
    for (int cpuId = 0; cpuId < numCPUs; ++cpuId) {
      SPDLOG_TRACE(logger::console(), "discover cpu {}", cpuId);
      auto cpu = std::make_shared<CPU>(cpuId);
      topology.cpus_.insert(std::make_pair(cpuId, cpu));

      // if NUMA is available, associate each cpu with its numa region

      if (numa::available()) {
        int numaId = numa::node_of_cpu(cpuId);
        SPDLOG_TRACE(logger::console(), "discover numa {} for cpu {}", numaId, cpuId);
        auto numa = topology.numas_[numaId];
        cpu->numa_ = numa;
        numa->cpus_.insert(cpu);
      }
    }

    // add gpus to topology
    for (auto gpu : detail::make_gpus()) {
      SPDLOG_TRACE(logger::console(), "discover gpu {}", gpu->cudaId_);
      topology.cudaGpus_.insert(std::make_pair(gpu->cudaId_, gpu));
      topology.nvmlGpus_.insert(std::make_pair(gpu->nvmlIdx_, gpu));
    }

    // check all CPUs
    for (auto kv : topology.cpus_) {
      unsigned int cpuId = kv.first;
      auto cpu = kv.second;
      // look up the nvml device index associated with this CPU
      unsigned int count = 0;
      NVML(nvmlSystemGetTopologyGpuSet(cpuId, &count, nullptr));
      std::vector<nvmlDevice_t> devices(count);
      NVML(nvmlSystemGetTopologyGpuSet(cpuId, &count, devices.data()));

      for (auto nvmlDevice : devices) {
        // get the GPU object
        unsigned int nvmlIdx;
        NVML(nvmlDeviceGetIndex(nvmlDevice, &nvmlIdx));
        auto gpu = topology.nvmlGpus_[nvmlIdx];

        SPDLOG_TRACE(logger::console(), "discover gpu {} has affinity with cpu {}", gpu->cudaId_, cpuId);

        if (cpu->numa_) {
          // the gpu is associated with all numa regions of the cpus it is associated with
          gpu->numas_.insert(cpu->numa_);

          // add the gpu to the NUMA region
          cpu->numa_->gpus_.insert(gpu);
        }

        // add the cpu to the gpu
        cpu->gpus_.insert(gpu);

        // add the gpu to the cpu
        gpu->cpus_.insert(cpu);
      }
    }
    init = true;
  }

  return topology;
}

} // namespace topology

} // namespace pangolin
