#pragma once

#include <utility>

#include "namespace.hpp"
#include "types.hpp"

namespace pangolin {

/*! A directed edge is a std::pair

    first() is src, second() is dst
*/
typedef std::pair<Uint, Uint> Edge;

/*! A directed edge is a std::pair

    first() is src, second() is dst
*/
template <typename NodeTy> using EdgeTy = std::pair<NodeTy, NodeTy>;

/*! A directed, weighted edge

    \tparam N the node at each end of the edge
    \tparam T the type of the edge's value
*/
template <typename N, typename T> struct WeightedDiEdge {
  N src;
  N dst;
  T val; //!< edge weight
};

template <typename N, typename T> struct DiEdge {
  N src;
  N dst;
};

} // namespace pangolin