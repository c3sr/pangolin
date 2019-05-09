#pragma once

#include "pangolin/edge.hpp"

namespace pangolin {

namespace generator {

/*! A directed complete graph, with N(N-1) edges.

    No nodes have self-edges
*/
template <typename Node> class Complete {

private:
  uint64_t numNodes_;

public:
  class iterator {
    friend class Complete;

  private:
    const uint64_t numNodes_;
    Node pos_; //<! flat position in 2D numNodes_ * numNodes_ space

    iterator(uint64_t numNodes, Node pos) : numNodes_(numNodes), pos_(pos) {
      // possible that the first edge should be skipped
      if (skip_edge()) {
        (*this)++;
      }
    }

    Node get_src() const { return 0 == numNodes_ ? 0 : pos_ / numNodes_; }
    Node get_dst() const { return 0 == numNodes_ ? 0 : pos_ % numNodes_; }
    bool done() const { return uint64_t(pos_) >= numNodes_ * numNodes_; }
    bool skip_edge() const {
      if (done()) {
        return false;
      }
      return get_src() == get_dst();
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
  };

  Complete(const uint64_t numNodes) : numNodes_(numNodes) {}

  iterator begin() const { return iterator(numNodes_, 0); }
  iterator end() const { return iterator(numNodes_, numNodes_ * numNodes_); }
};

} // namespace generator

} // namespace pangolin