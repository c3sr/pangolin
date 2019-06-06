#pragma once

#include "spdlog/sinks/stdout_color_sinks.h"

namespace pangolin {
/*! initialize pangolin
 */
void init() {
  static bool init_ = false;
  if (init_)
    return;

  // create a logger and implicitly register it
  spdlog::stderr_color_st("console");

  // don't init again if init() called twice
  init_ = true;
}

} // namespace pangolin
