#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#if PANGOLIN_USE_MPI == 1
#include <mpi.h>
#endif

#include "pangolin/init.hpp"

using namespace pangolin;

TEST_CASE("available") {
  pangolin::init();
#if PANGOLIN_USE_MPI == 1
  MPI_Init(NULL, NULL);
  MPI_Finalize();
#else
  INFO("mpi not installed");
  REQUIRE(false);
#endif
}
