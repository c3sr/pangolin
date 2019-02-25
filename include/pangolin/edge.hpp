#pragma once

#include <utility>

#include "namespace.hpp"
#include "types.hpp"

PANGOLIN_BEGIN_NAMESPACE()

/*! A directed edge is a std::pair

    first() is src, second() is dst
*/
typedef std::pair<Uint, Uint> Edge;

PANGOLIN_END_NAMESPACE()