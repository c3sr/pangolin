#include "pangolin/logger.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"

PANGOLIN_BEGIN_NAMESPACE();

std::shared_ptr<spdlog::logger> logger::console =
    spdlog::stderr_color_mt("pangolin");

namespace logger {

void set_level(const Level &level) {
  switch (level) {
  case TRACE: {
    console->set_level(spdlog::level::trace);
    return;
  }
  case DEBUG: {
    console->set_level(spdlog::level::debug);
    return;
  }
  case INFO: {
    console->set_level(spdlog::level::info);
    return;
  }
  case WARN: {
    console->set_level(spdlog::level::warn);
    return;
  }
  case ERR: {
    console->set_level(spdlog::level::err);
    return;
  }
  case CRITICAL: {
    console->set_level(spdlog::level::critical);
    return;
  }
  }
}
} // namespace logger

PANGOLIN_END_NAMESPACE()