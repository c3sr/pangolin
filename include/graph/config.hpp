#pragma once

#include <string>

struct Config {
    std::string type_;
    int numCPUThreads_ = 1;
    int numGPUs_ = 1;
};