#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"

using namespace pangolin;

// defined in test_logger_helper.cpp
void log_a_message();

TEST_CASE("0") {
  pangolin::init();
  pangolin::logger::set_level(logger::Level::DEBUG);
  log_a_message();
  LOG(info, "this is a different message");
}
