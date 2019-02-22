
#pragma once

#include <spdlog/spdlog.h>

namespace logger
{
extern std::shared_ptr<spdlog::logger> console;

enum Level {TRACE, DEBUG, INFO, WARN, ERR, CRITICAL};
void set_level(const Level &level);

} // namespace logger

#define LOG(level, ...) logger::console->level(__VA_ARGS__)
