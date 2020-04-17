#pragma once

/*!

pangolin wrapper around libnuma

If pangolin is not compiled with NUMA or NUMA is not available, the functions create a fake NUMA region 0 that contains
all CPUs.

It should otherwise be consistent with libnuma.

*/

#include <cassert>
#include <set>

#if USE_NUMA == 1
#include <numa.h>
#include <numaif.h>
#endif // USE_NUMA == 1

#include <errno.h>

#include "logger.hpp"

namespace pangolin {

namespace numa {

/*! return true if pangolin is numa-aware.

    \return true if numa is available and pangolin is compiled with NUMA support. False otherwise
*/
inline bool available() {
#if USE_NUMA == 1
  return -1 == ::numa_available() ? false : true;
#else  // USE_NUMA == 1
  SPDLOG_TRACE(logger::console(), "USE_NUMA not defined. numa_available() = false");
  return false;
#endif // USE_NUMA == 1
}

/*! call numa_num_configured_cpus()

    return 0 if NUMA is not available
*/
inline int num_configured_cpus() {
#if USE_NUMA == 1
  if (available()) {
    return ::numa_num_configured_cpus();
  } else {
    LOG(error, "numa not available in {}", __PRETTY_FUNCTION__);
    return 0;
  }
#else
  LOG(warn, "USE_NUMA not defined in {}", __PRETTY_FUNCTION__);
  return 0;
#endif
}

/*! return the node that cpu is in

    returns -1 if it cannot be determined
*/
int node_of_cpu(const int cpu) {
#if USE_NUMA == 1
  if (available()) {
    return ::numa_node_of_cpu(cpu);
  }
#endif // USE_NUMA
  return -1;
}

/*! return the numa node that the page of ptr is allocated in.
  If NUMA is not supported or otherwise it cannot be determined, return -1
  */
int node_of_addr(void *ptr, size_t pageSize) {

  assert(pageSize);

#if USE_NUMA == 1
  // round down to pageSize
  void *alignedPtr = reinterpret_cast<void *>((uintptr_t(ptr) / pageSize) * pageSize);
  int status[1];
  status[0] = -1;
  const int ret = move_pages(0 /*calling process*/, 1 /*1 page*/, &alignedPtr,
                             NULL /* don't move, report page location*/, status, 0 /*no flags*/);
  if (0 != ret) {
    LOG(error, "error in move_pages");
    return -1;
  } else {
    if (-EFAULT == *status) {
      LOG(error, "{:x} is in a zero page or memory area is not mapped", uintptr_t(ptr));
      return -1;
    } else if (-ENOENT == *status) {
      LOG(error, "the page containing {:x} is not present", uintptr_t(ptr));
      return -1;
    } else if (*status < 0) {
      LOG(error, "status was {}", *status);
      return -1;
    } else {
      return *status;
    }
  }
#else  // USE_NUMA == 1
  return -1;
#endif // USE_NUMA == 1
}

/*! \brief cause errors if NUMA is not able to follow instructions

    Does nothing if pangolin::numa::numa_available() is false
    calls numa_set_strict(1), numa_set_bind_policy(1), numa_exit_on_warn=1 ,numa_exit_on_error=1
*/
inline void set_strict() {
#if USE_NUMA == 1
  if (available()) {
    numa_set_strict(1);
    LOG(debug, "set numa_set_strict(1)");
    numa_set_bind_policy(1);
    LOG(debug, "set numa_set_bind_policy(1)");

    numa_exit_on_warn = 1;
    LOG(debug, "set numa_exit_on_warn = 1");
    numa_exit_on_error = 1;
    LOG(debug, "set numa_exit_on_error = 1");
  } else {
    LOG(error, "numa not available in {}", __PRETTY_FUNCTION__);
  }
#else  // USE_NUMA == 1
  LOG(debug, "USE_NUMA not defined");
#endif // USE_NUMA == 1
}

/*! \brief bind execution and allocation to node

    Does nothing if pangolin::numa::numa_available() is false
    Uses numa_bind()
*/
inline void bind(const int node //!< NUMA node to bind to
) {
#if USE_NUMA == 1
  if (available()) {
    if (-1 == node) {
      numa_bind(numa_all_nodes_ptr);
    } else if (node >= 0) {
      struct bitmask *nodemask = numa_allocate_nodemask();
      nodemask = numa_bitmask_setbit(nodemask, node);
      numa_bind(nodemask);
      numa_free_nodemask(nodemask);
    } else {
      LOG(error, "expected node >= -1");
    }
  } else {
    LOG(error, "numa not available in {}", __PRETTY_FUNCTION__);
  }
#else  // USE_NUMA == 1
  LOG(debug, "USE_NUMA not defined");
#endif // USE_NUMA == 1
}

/*! Bind future allocation to a numa node
 */
inline void membind(const int node //!< NUMA node to bind to
) {
#if USE_NUMA == 1
  if (available()) {
    if (-1 == node) {
      numa_set_membind(numa_all_nodes_ptr);
    } else if (node >= 0) {
      struct bitmask *nodemask = numa_allocate_nodemask();
      nodemask = numa_bitmask_setbit(nodemask, node);
      numa_set_membind(nodemask);
      numa_free_nodemask(nodemask);
    } else {
      LOG(error, "numa not available in {}", __PRETTY_FUNCTION__);
    }
  }
#else  // USE_NUMA == 1
  LOG(debug, "USE_NUMA not defined in {}", __PRETTY_FUNCTION__);
#endif // USE_NUMA == 1
}

/*! \brief bind execution and allocation to all nodes

    Does nothing if pangolin::numa::numa_available() is false
    Uses numa_bind(numa_all_nodes_ptr)
*/
inline void unbind() {
#if USE_NUMA == 1
  if (available()) {
    numa_bind(numa_all_nodes_ptr);
  } else {
    LOG(error, "numa not available in {}", __PRETTY_FUNCTION__);
  }
#else  // USE_NUMA == 1
  LOG(debug, "USE_NUMA not defined");
#endif // USE_NUMA == 1
}

/*! all nodes on which the calling task may allocate memory.

\return a set of node ids. Empty if NUMA not available
 */
inline std::set<int> all_nodes() {
  std::set<int> numas;
#if USE_NUMA == 1
  if (available()) {
    for (int i = 0; i < ::numa_num_possible_nodes(); ++i) {
      if (::numa_bitmask_isbitset(numa_all_nodes_ptr, i)) {
        numas.insert(i);
      }
    }
  }
#endif
  return numas;
}

namespace detail {} // namespace detail

} // namespace numa

} // namespace pangolin