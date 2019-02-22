#pragma once

#include <utility>

#include "types.hpp"
#include "namespace.hpp"

PANGOLIN_BEGIN_NAMESPACE()

/*! A directed edge is a std::pair

    first() is src, second() is dst
*/
typedef std::pair<Uint, Uint> Edge;

PANGOLIN_END_NAMESPACE()