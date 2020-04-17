#pragma once

#include <vector>

#include "edge.hpp"
namespace pangolin {


template <typename T>
using DiEdgeList = std::vector<DiEdge<T>>;

} // namespace pangolin
