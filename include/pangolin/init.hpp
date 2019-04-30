#pragma once

#include "spdlog/sinks/stdout_color_sinks.h"

namespace pangolin {
/*! initialize pangolin
 */
void init() {
  // create a logger and implicitly register it
  spdlog::stdout_color_mt("console");
}
} // namespace pangolin