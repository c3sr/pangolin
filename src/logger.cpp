#include "graph/logger.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"

std::shared_ptr<spdlog::logger> logger::console = spdlog::stderr_color_mt("tri");