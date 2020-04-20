#pragma once

#ifdef PANGOLIN_USE_CUSPARSE

#include <cusparse.h>

#include "pangolin/logger.hpp"

namespace pangolin {

#define CUSPARSE(stmt) checkCusparse(stmt, __FILE__, __LINE__);

static void checkCusparse(cusparseStatus_t result, const char *file, const int line) {
  if (result != CUSPARSE_STATUS_SUCCESS) {
    printf("cusparse Error: %s in %s : %d\n", cusparseGetErrorString(result),
           file, line);
    exit(-1);
  }
}

} // namespace pangolin

#endif
