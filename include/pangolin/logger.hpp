
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace pangolin {

namespace logger {
std::shared_ptr<spdlog::logger> console = spdlog::stderr_color_mt("pangolin");


/*!
Log levels used by pangolin's internal logger.

Trace-level messages are disabled during release builds.
*/
enum Level { TRACE, DEBUG, INFO, WARN, ERR, CRITICAL };


/*! Set the log level of pangolin's internal logger
 
*/
inline void set_level(const Level &level) {
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

} // namespace pangolin

#define LOG(level, ...) pangolin::logger::console->level(__VA_ARGS__)
