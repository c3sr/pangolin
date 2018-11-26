#pragma once

#include <string>

struct Config
{
    std::string type_ = "gpu";
    int numCPUThreads_ = 0;
    int numGPUs_ = 1;
};