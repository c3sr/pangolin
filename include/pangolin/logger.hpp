
#pragma once

#include "namespace.hpp"
#include <spdlog/spdlog.h>

PANGOLIN_BEGIN_NAMESPACE()

namespace logger {
extern std::shared_ptr<spdlog::logger> console;

/*!
Log levels used by pangolin's internal logger.

Trace-level messages are disabled during release builds.
*/
enum Level { TRACE, DEBUG, INFO, WARN, ERR, CRITICAL };
void set_level(const Level &level);

} // namespace logger

PANGOLIN_END_NAMESPACE()

#define LOG(level, ...) pangolin::logger::console->level(__VA_ARGS__)
