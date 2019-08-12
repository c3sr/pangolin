#pragma once

#include <vector>
#include <atomic>

#include "logger.hpp"


// FIXME:
// could the consumer ready be signaled by an empty consumer buffer?


namespace pangolin {

/*! 

    \tparam N the number of buffer entries
    \tparam T the type of the entry
*/
template <typename T, size_t N = 512> struct DoubleBuffer {

  typedef T value_type;

private:


  std::atomic<bool> producer_ready_;
  std::atomic<bool> consumer_ready_;

  std::mutex mtx_;
  std::condition_variable consumerBusy; //!< block producer from flipping
  std::condition_variable producerBusy; //!< block consumers until flipped
  
  std::atomic<bool> close_; //!< true if new values will never be added to the queue

public:
  std::vector<T> produce;
  std::vector<T> consume;

  DoubleBuffer() : producer_ready_(false), consumer_ready_(false), close_(false) {
    produce.reserve(N);
    consume.reserve(N);
  }

  void flip() {
    std::swap(produce, consume);
    produce.clear(); // produce buffer should now be empty
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

/*



 */


  void notify_consumer() {
    producer_ready_ = true; // the producer has prepared the consumer buffer
    producerBusy.notify_all(); // release anyone waitin on the producer
  }

  void wait_consumer() {
    std::unique_lock<std::mutex> lock(mtx_);

    // wait for the consumer to signal he's done with the buffer
    consumerBusy.wait(lock, [this]() { return consumer_ready_.load(); });

    // once the consumer says he's done, clear it so if we ask again we have to wait again
    consumer_ready_ = false;

    lock.unlock();
  }

/*! let the producer know the consumer is done
 */
  void notify_producer() {
    consume.clear(); // consumer is done with this buffer
    consumer_ready_ = true; // consumer is ready for more stuff
    consumerBusy.notify_all(); // release anyone waiting on the consumer
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

  bool is_closed() const { return close_.load(); }


};

} // namespace pangolin