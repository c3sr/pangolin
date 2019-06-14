#pragma once

#include <vector>

#include "logger.hpp"

namespace pangolin {

/*! 

    \tparam N the number of buffer entries
    \tparam T the type of the entry
*/
template <typename T, size_t N = 512> struct DoubleBuffer {

  typedef T value_type;

private:


  volatile bool producer_ready_;
  volatile bool consumer_ready_;

  std::mutex mtx_;
  std::condition_variable consumerBusy; //<! block producer from flipping
  std::condition_variable producerBusy; //<! block consumers until flipped
  
  volatile bool close_; //!< true if new values will never be added to the queue

public:
  std::vector<T> produce;
  std::vector<T> consume;

  DoubleBuffer() : close_(false) {
    produce.reserve(N);
    consume.reserve(N);
  }

  void flip() {
    std::swap(produce, consume);
    produce.clear();
    producer_ready_ = false;
    consumer_ready_ = false;
  }

  void wait_producer() {
    std::unique_lock<std::mutex> lock(mtx_);

    // wait for the producer to signal the buffer is okay to use, or if the buffer is closed
    // LOG(debug, "in wait_producer, {}, {}", producer_ready_, is_closed());
    producerBusy.wait(lock, [this]() { return producer_ready_ || is_closed(); });

    // once the producer has said we're good, clear it so we have to wait for the producer again
    producer_ready_ = false;

    lock.unlock();
  }

  void notify_consumer() {
    producer_ready_ = true;
    producerBusy.notify_all();
  }

  void wait_consumer() {
    std::unique_lock<std::mutex> lock(mtx_);

    // wait for the consumer to signal he's done with the buffer
    consumerBusy.wait(lock, [this]() { return consumer_ready_; });

    // once the consumer says he's done, clear it so if we ask again we have to wait again
    consumer_ready_ = false;

    lock.unlock();
  }

  void notify_producer() {
    consumer_ready_ = true;
    consumerBusy.notify_all();
  }

  size_t consume_size() {
    return consume.size();
  }

  /*! \brief Inform the DoubleBuffer that no new entries will be added

  */
  void close() {
    close_ = true;
    // wake up the consumer who might be waiting for the queue
    producerBusy.notify_all();
  }

  bool is_closed() const { return close_; }


};

} // namespace pangolin