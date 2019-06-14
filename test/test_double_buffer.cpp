#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <thread>

#include "pangolin/double_buffer.hpp"
#include "pangolin/init.hpp"

using namespace pangolin;

template<typename T>
void consumer(DoubleBuffer<T> &db) {

  size_t consumed_count = 0;
  while (true) {



    LOG(debug, "consumer: waiting for producer's okay");
    db.wait_producer(); // wait for producer to notify us
    // if the buffer is closed, we shouldn't expect to see anything else
    if (db.is_closed()) {
      break;
    }

    LOG(debug, "consumer: i'm consuming!");

    // consume anything the producer put in the buffer
    for (size_t i = 0; i < db.consume_size(); ++i) {
      T val = db.consume[i];
      (void)val; // throw away
      consumed_count++;
    }


    
    // LOG(debug, "consumer: consumed {} values", consumed_count);
    LOG(debug, "consumer: done consuming. notifying producer...");
    db.notify_producer(); // tell the producer we're ready
    LOG(debug, "consumer: told producer");


  }
  LOG(debug, "consumer: consumed {} values", consumed_count);

}

template<typename T>
void producer(DoubleBuffer<T> &db) {

  for (size_t b = 0; b < 10; ++b) {
    LOG(debug, "producer starting batch {}", b);

    LOG(debug, "producer notifying consumer");
    db.notify_consumer(); // let the consumer go even if there is nothing in the buffer

    std::chrono::milliseconds timespan(100);
    std::this_thread::sleep_for(timespan);

    for (size_t i = 0; i < 10; ++i) {
      db.produce.push_back(i);
    }
    LOG(debug, "producer added stuff");

    LOG(debug, "producer waiting for consumer");
    db.wait_consumer(); // wait for consumer to be done

    LOG(debug, "producer flipping buffer");
    db.flip(); // shouldn't this always notify the consumer?
  }

  db.notify_consumer();

  db.wait_consumer(); // wait for the consumer to finish
  LOG(debug, "producer closing buffer");
  db.close();
}


TEMPLATE_TEST_CASE("buffer", "[gpu]", int, size_t) {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);

  DoubleBuffer<TestType> db;

  LOG(debug, "started consumer thread");
  auto c = std::thread(consumer<TestType>, std::ref(db));

  LOG(debug, "main thread as producer");
  producer<TestType>(db);

  LOG(debug, "waiting for consumer to join");
  c.join();
}
  
