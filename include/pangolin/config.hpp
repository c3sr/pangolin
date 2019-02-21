#pragma once

#include <string>
#include <vector>

#include "namespace.hpp"

PANGOLIN_BEGIN_NAMESPACE()

struct Config
{
    std::string type_;
    int numCPUThreads_ = 0;
    std::vector<int> gpus_; // which GPUs to use. duplicate entries will be treated as two different GPUs
    unsigned int seed_;
    std::string storage_;
    std::string kernel_; // which kernel to use, if there are multiple choices
    bool hints_ = 0;         // use unified memory hints
};

PANGOLIN_END_NAMESPACE()