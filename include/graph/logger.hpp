
#pragma once

#include "spdlog/spdlog.h"


  namespace logger {
    extern std::shared_ptr<spdlog::logger> console;
  } // namespace logger

#define LOG(level, ...) logger::console->level(__VA_ARGS__)