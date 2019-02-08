#pragma once

#include <limits>
#include <cassert>
#include <cstdint>

#include "logger.hpp"


static int checked_narrow(const uint64_t u) {
    if (u > uint64_t(std::numeric_limits<int>::max())){
        LOG(critical, "{} is too large for conversion to int", u);
        exit(-1);
    }
    return static_cast<int>(u);
}