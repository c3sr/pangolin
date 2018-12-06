#pragma once

#include <string>
#include <vector>

struct Config
{
    std::string type_;
    int numCPUThreads_ = 0;
    std::vector<int> gpus_; // which GPUs to use. duplicate entries will be treated as two different GPUs
    unsigned int seed_;
    std::string storage_;
};