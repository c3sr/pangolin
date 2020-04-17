#pragma once

#include <utility>

#include "namespace.hpp"

namespace pangolin {

/*! A directed, weighted edge

    \tparam N the node at each end of the edge
    \tparam T the type of the edge's value
*/
template <typename Node, typename T> struct WeightedDiEdge {
  Node src;
  Node dst;
  T val; //!< edge weight
};

template <typename Node> struct DiEdge {
  Node src;
  Node dst;

  DiEdge(Node _src, Node _dst) : src(_src), dst(_dst) {}

  DiEdge() = default;
};

} // namespace pangolin