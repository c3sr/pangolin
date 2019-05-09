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
    Node src_;
    Node dst_;
    bool done_;

    iterator(uint64_t numNodes, Node src, Node dst) : numNodes_(numNodes), src_(src), dst_(dst) {
      done_ = (static_cast<uint64_t>(src) >= numNodes) || (static_cast<uint64_t>(dst) >= numNodes);
    }

  public:
    const EdgeTy<Node> operator*() const {
      assert(src_ < numNodes_);
      assert(dst_ < numNodes_);
      assert(!done_);
      return EdgeTy<Node>(src_, dst_);
    }

    iterator &operator++() // ++prefix
    {
      if (!done_) {
        ++dst_;
        if (static_cast<uint64_t>(dst_) >= numNodes_) {
          dst_ = 0;
          ++src_;
        }
        if (static_cast<uint64_t>(src_) >= numNodes_) {
          done_ = true;
        }
      }

      if (!done_) {
        if (dst_ == src_) {
          ++*this;
        }
      }

      return *this;
    }

    iterator operator++(int) // postfix++
    {
      iterator i(*this);
      ++(*this);
      return i;
    }

    bool operator==(const iterator &rhs) const {
      if (done_ && rhs.done_) {
        return true;
      } else if (done_ ^ rhs.done_) { // one only one is done, not equal
        return false;
      } else {
        return (src_ == rhs.src_) && (dst_ == rhs.dst_);
      }
    }
    bool operator!=(const iterator &rhs) const { return !((*this) == rhs); }
  };

  Complete(const uint64_t numNodes) : numNodes_(numNodes) {}

  iterator begin() const { return iterator(numNodes_, 0, 1); }
  iterator end() const { return iterator(numNodes_, numNodes_, numNodes_); }
};

} // namespace generator

} // namespace pangolin