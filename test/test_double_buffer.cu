#include <catch2/catch.hpp>

#include <thread>

#include "pangolin/double_buffer.hpp"
#include "pangolin/init.hpp"

using namespace pangolin;

template<typename T>
void consumer(size_t &consumedCount, DoubleBuffer<T> &db, const size_t wait_ms = 0) {

  consumedCount = 0;
  while (true) {

    LOG(debug, "consumer: waiting for producer's okay");
    db.wait_producer(); // wait for producer to notify us

    LOG(debug, "consumer: i'm consuming!");

    // consume anything the producer put in the buffer
    for (size_t i = 0; i < db.consume_size(); ++i) {
      T val = db.consume[i];
      (void)val; // throw away
      std::chrono::milliseconds timespan(wait_ms);
      std::this_thread::sleep_for(timespan);
      consumedCount++;
    }
    
    LOG(debug, "consumer: done consuming ({} so far)", consumedCount);

    // if the buffer is closed, nothing more will be coming, so we are done consuming
    if (db.is_closed()) {
      LOG(debug, "consumer: saw buffer was closed.");
      break;
    }

    LOG(debug, "consumer: notifying producer...");
    db.notify_producer(); // tell the producer we're ready

  }
  LOG(debug, "consumer: exiting");

}

template<typename T>
void producer(size_t &producedCount, DoubleBuffer<T> &db, const size_t wait_ms = 0) {

  producedCount = 0;
  for (size_t b = 0; b < 10; ++b) {
    LOG(debug, "producer starting batch {}", b);

    LOG(debug, "producer notifying consumer");
    db.notify_consumer(); // first iteration there will be nothing in the buffer, but that's okay

    REQUIRE(db.produce.empty());
    for (size_t i = 0; i < 10; ++i) {
      std::chrono::milliseconds timespan(wait_ms);
      std::this_thread::sleep_for(timespan);
      db.produce.push_back(T(i));
      ++producedCount;
    }
    LOG(debug, "producer: added stuff. waiting for consumer.");
    db.wait_consumer(); // wait for consumer to be done

    LOG(debug, "producer: consumer done. flipping buffer");
    db.flip(); // shouldn't this always notify the consumer?
  }

  // let the consumer consume the last things we added
  LOG(debug, "producer: final consumer notification");
  db.notify_consumer();
  LOG(debug, "producer: waiting for consumer");
  db.wait_consumer();

  LOG(debug, "producer: consumer done. closing buffer");
  db.close();
}

TEMPLATE_TEST_CASE("empty buffer", "[gpu]", int, size_t) {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);

  DoubleBuffer<TestType> db;
  size_t consumedCount;

  LOG(debug, "started consumer thread");
  auto c = std::thread(consumer<TestType>, std::ref(consumedCount), std::ref(db), 0);

  db.close();

  LOG(debug, "waiting for consumer to join");
  c.join();

  REQUIRE(consumedCount == 0);
}


TEMPLATE_TEST_CASE("slow consumer", "[gpu]", int, size_t) {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);

  DoubleBuffer<TestType> db;
  size_t consumedCount, producedCount;

  LOG(debug, "started consumer thread");
  auto c = std::thread(consumer<TestType>, std::ref(consumedCount), std::ref(db), 10);

  LOG(debug, "main thread as producer");
  producer<TestType>(producedCount, db, 0);

  LOG(debug, "waiting for consumer to join");
  c.join();

  REQUIRE(consumedCount == producedCount);
}
  
TEMPLATE_TEST_CASE("slow producer", "[gpu]", int, size_t) {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);

  DoubleBuffer<TestType> db;
  size_t consumedCount, producedCount;

  LOG(debug, "started consumer thread");
  auto c = std::thread(consumer<TestType>, std::ref(consumedCount), std::ref(db), 0);

  LOG(debug, "main thread as producer");
  producer<TestType>(producedCount, db, 10);

  LOG(debug, "waiting for consumer to join");
  c.join();

  REQUIRE(consumedCount == producedCount);
}

TEMPLATE_TEST_CASE("full speed", "[gpu]", int, size_t) {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);

  DoubleBuffer<TestType> db;
  size_t consumedCount, producedCount;

  LOG(debug, "started consumer thread");
  auto c = std::thread(consumer<TestType>, std::ref(consumedCount), std::ref(db), 0);

  LOG(debug, "main thread as producer");
  producer<TestType>(producedCount, db, 0);

  LOG(debug, "waiting for consumer to join");
  c.join();

  REQUIRE(producedCount == consumedCount);
}