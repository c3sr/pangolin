#pragma once

#include <array>
#include <deque>

#include "logger.hpp"

namespace pangolin {

/*! A circular buffer of bounded size

    \tparam N the number of buffer entries
    \tparam T the type of the entry
*/
template <typename T, size_t N = 512> struct BoundedBuffer {

  typedef T value_type;

private:
  size_t head_;
  size_t tail_;
  size_t count_;
  std::mutex mtx_;
  std::condition_variable notFull;  //!< block pushers until queue is not full
  std::condition_variable notEmpty; //!< block poppers until queue is not empty
  std::array<T, N> buffer_;
  bool close_; //!< true if new values will never be added to the queue

public:
  BoundedBuffer() : close_(false), head_(0), tail_(0), count_(0) {}

  /*! move ctor
   */
  BoundedBuffer(BoundedBuffer &&other)
      : close_(other.close_), head_(other.head_), tail_(other.tail_), count_(other.count_) {
    buffer_ = std::move(other.buffer_);
    other.close_ = true;
    other.head_ = 0;
    other.tail_ = 0;
    other.count_ = 0;
  }

  /*! \brief Add as many values as possible from the front of vals to the buffer and return

    Will block until at least one element is added.
   */
  void push_some(std::deque<T> &vals //!< [inout] the source of values to add to
                                     //!< the BoundedBuffer
  ) {
    assert(!closed());
    std::unique_lock<std::mutex> lock(mtx_);

    // wait for the buffer to not be full
    notFull.wait(lock, [this]() { return !full(); });

    assert(!full());
    assert(!closed());

    // add entries from the front of vals
    // we will remove entries from vals after releasing the lock
    const size_t numToAdd = std::min(N - count(), vals.size());
    for (size_t i = 0; i < numToAdd; ++i) {
      buffer_[head_] = std::move(vals[i]);
      advance_head();
    }

    lock.unlock();

    // release anyone waiting on the buffer to not be empty
    notEmpty.notify_all();

    // pop the values off the front of the queue before returning
    for (size_t i = 0; i < numToAdd; ++i) {
      vals.pop_front();
    }

    // SPDLOG_DEBUG(pangolin::logger::console, "pushed {}", numToAdd);
    return;
  }

  /*! \brief remove as many values from the BoundedBuffer as possible and return

    If the buffer is not closed, blocks until there is at least one entry in the
    buffer and then returns the buffer entries.
    If the buffer is closed, zero entries are returned.
  */
  std::vector<T> pop_some() {
    std::vector<T> vals;
    // before acquiring the lock, reserve space for possibly removing all
    // elements from the buffer
    vals.reserve(N);

    std::unique_lock<std::mutex> lock(mtx_);

    // wait until the buffer is not empty or it is closed
    notEmpty.wait(lock, [this]() { return (!empty()) || close_; });

    // if the buffer is closed, this could add no elements to vals
    while (!empty()) {
      vals.push_back(std::move(buffer_[tail_]));
      advance_tail();
    }

    lock.unlock();

    // wake anyone waiting on the buffer to not be full
    notFull.notify_all();
    // SPDLOG_DEBUG(pangolin::logger::console, "popped {}", vals.size());
    return vals;
  }

  /*! \brief Inform the BoundedBuffer that no new entries will be added

  */
  void close() {
    close_ = true;
    // wake up everyone who might be trying to pop from the queue
    notEmpty.notify_all();
  }

  bool empty() const { return count_ == 0; }
  bool full() const { return count_ >= N; }
  size_t count() const { return count_; }
  bool closed() const { return close_; }

private:
  void advance_head() {
    head_ = (head_ + 1) % N;
    ++count_;
  }
  void advance_tail() {
    tail_ = (tail_ + 1) % N;
    --count_;
  }
};

} // namespace pangolin