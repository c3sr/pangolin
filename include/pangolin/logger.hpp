
#pragma once

#include "spdlog/spdlog.h"

namespace logger
{
extern std::shared_ptr<spdlog::logger> console;
} // namespace logger

#define LOG(level, ...) logger::console->level(__VA_ARGS__)

// if NDEBUG is defined (during release builds), TRACE is a no-op
#ifndef NDEBUG
#define TRACE(...)                           \
    do                                       \
    {                                        \
        logger::console->trace(__VA_ARGS__); \
    } while (0)
#else
#define TRACE(...)
#endif