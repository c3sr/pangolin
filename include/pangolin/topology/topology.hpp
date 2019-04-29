#pragma once

#include <set>
#include <climits>

#include <nvml.h>

#include "pangolin/numa.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {
namespace topology {

bool _init = false;

void _lazy_init() {
  if (!_init) {
    NVML(nvmlInit());
    _init = true;
  }
}

/*! get the cpus with affinity for gpu

    \return a set<int> of cpus that have affinity with gpu
 */
inline std::set<int> device_cpu_affinity(const int gpu //<! the gpu to get CPU affinity for
) {
  _lazy_init();
  nvmlDevice_t nvmlDev = nullptr;
  NVML(nvmlDeviceGetHandleByIndex(gpu, &nvmlDev));

  unsigned int cpuSetSize = static_cast<unsigned int>((std::thread::hardware_concurrency() + sizeof(unsigned long) - 1) / sizeof(unsigned long));
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


} // namespace topology
} // namespace pangolin