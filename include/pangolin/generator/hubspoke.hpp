#pragma once

#include "pangolin/edge.hpp"

namespace pangolin {

namespace generator {

/*! A hub and spoke graph with N spokes

    No nodes have self-edges

    Nodes n in 0...N-1 are all connected to n-1 and n + 1 (except for node 0 and n-1). Node n is connected to each of
   them

   Keep src < dst for a high in-degree hub.
   Keep src > dst for a high out-degree hub.
*/
template <typename Node> class HubSpoke {

private:
  uint64_t numSpokes_;

public:
  /*! get the number of unique triangles in this graph
   */
  uint64_t num_triangles() const {
    if (numSpokes_ < 1) {
      return 0;
    } else {
      return numSpokes_ - 1;
    }
  }

  class iterator {
    friend class HubSpoke;

  private:
    const uint64_t N;
    Node pos_; //!< check whether all pairs of edges should exist

    iterator(uint64_t n, Node pos) : N(n), pos_(pos) {
      if (skip_edge()) {
        (*this)++;
      }
    }

    // defined to order by src
    Node get_src() const { return 0 == N ? 0 : pos_ / N; }
    Node get_dst() const { return 0 == N ? 0 : pos_ % N; }

    bool done() const { return uint64_t(pos_) >= N * N; }
    bool skip_edge() const {
      if (done()) {
        return false;
      }

      const Node src = get_src();
      const Node dst = get_dst();

      // if either src or dst is the hub, keep the edge
      if ((uint64_t(dst + 1) == N) ^ (uint64_t(src + 1) == N)) {
        return false;
      }

      // keep 0-1
      if (src == 0) {
        if (dst == 1) {
          return false;
        }
      } else if (uint64_t(src) + 1 < N) { // otherwise keep n-1 -> n -> n+1
        if (dst + 1 == src || src + 1 == dst) {
          return false;
        }
      }

      return true;
    }

  public:
    const EdgeTy<Node> operator*() const {
      assert(!done());
      return EdgeTy<Node>(get_src(), get_dst());
    }

    iterator &operator++() // ++prefix
    {
      do {
        ++pos_;
      } while (skip_edge());
      return *this;
    }

    iterator operator++(int) // postfix++
    {
      iterator i(*this);
      ++(*this);
      return i;
    }

    bool operator==(const iterator &rhs) const {
      if (done() && rhs.done()) {
        return true;
      } else if (done() ^ rhs.done()) { // one only one is done, not equal
        return false;
      } else {
        return pos_ == rhs.pos_;
      }
    }
    bool operator!=(const iterator &rhs) const { return !((*this) == rhs); }
  }; // namespace generator

  HubSpoke(const uint64_t numSpokes) : numSpokes_(numSpokes) {}

  iterator begin() const { return iterator(numSpokes_ + 1, 0); }
  iterator end() const { return iterator(numSpokes_ + 1, (numSpokes_ + 1) * (numSpokes_ + 1)); }
}; // namespace pangolin

} // namespace generator

} // namespace pangolin