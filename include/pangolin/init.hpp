#pragma once

#include <cstdio>

#include "spdlog/sinks/stdout_color_sinks.h"

#include "topology/topology.hpp"

namespace pangolin {
/*! initialize pangolin
 */
inline void init() {
  static bool init_ = false;
  if (init_) {
    fprintf(stderr, "WARN: init called more than once\n");
    return;
  }

  // create a logger and implicitly register it
  spdlog::stderr_color_st("console");

  // implicitly create topology, to speed it up if it's called in the future
  topology::get();

  // don't init again if init() called twice
  init_ = true;
}

} // namespace pangolin
