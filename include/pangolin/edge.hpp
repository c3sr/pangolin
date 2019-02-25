#pragma once

#include <utility>

#include "namespace.hpp"
#include "types.hpp"

namespace pangolin {

/*! A directed edge is a std::pair

    first() is src, second() is dst
*/
typedef std::pair<Uint, Uint> Edge;

} // namespace pangolin