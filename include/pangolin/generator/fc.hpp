#pragma once

#include "pangolin/edge.hpp"

namespace pangolin {

namespace generator {

template <typename Node> class Complete {

private:
  size_t numNodes_;

public:
  class iterator {
    friend class Complete;

  private:
    const size_t numNodes_;
    Node src_;
    Node dst_;
    bool done_;

    iterator(size_t numNodes, Node src, Node dst)
        : numNodes_(numNodes), src_(src), dst_(dst) {
      done_ = (src < numNodes) && (dst < numNodes);
    }

  public:
    const EdgeTy<Node> operator*() const { return EdgeTy<Node>(src_, dst_); }

    iterator &operator++() // ++prefix
    {
      if (dst_ < numNodes_) {
        ++dst_;
      } else {
        if (src_ < numNodes_) {
          dst_ = 0;
          ++src_;
        } else {
          done_ = true;
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
  };

  Complete(size_t numNodes) : numNodes_(numNodes) {}

  iterator begin() const { return iterator(numNodes_, 0, 0); }
  iterator end() const { return iterator(numNodes_, numNodes_, numNodes_); }
};

} // namespace generator

} // namespace pangolin