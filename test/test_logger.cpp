#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"

using namespace pangolin;

// defined in test_logger_helper.cpp
void log_a_message();

TEST_CASE("logger") {
  pangolin::init();

  SECTION("messages from different compilation units") {
    log_a_message();
    LOG(info, "this is a different message");
  }

  SECTION("set_level") {
    pangolin::logger::set_level(logger::Level::DEBUG);
    LOG(debug, "a debug message");
  }
}
