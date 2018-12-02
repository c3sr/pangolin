#pragma once

#include <string>
#include <vector>

struct Config
{
    std::string type_ = "um";
    int numCPUThreads_ = 0;
    std::vector<int> gpus_; // which GPUs to use. duplicate entries will be treated as two different GPUs
    int numGPUs_ = 1;
    unsigned int seed_;
};