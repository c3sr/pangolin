#pragma once

#include <string>

struct Config
{
    std::string type_ = "um";
    int numCPUThreads_ = 0;
    int numGPUs_ = 1;
    unsigned int seed_;
};