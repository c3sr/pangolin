#pragma once

#include <string>

struct Config
{
    std::string type_ = "gpu";
    int numCPUThreads_ = 1;
    int numGPUs_ = 1;
};