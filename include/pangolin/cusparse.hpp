#pragma once

#include <cusparse.h>

#include "pangolin/logger.hpp"

#define CUSPARSE(stmt) checkCusparse(stmt, __FILE__, __LINE__);
void checkCusparse(cusparseStatus_t result, const char *file, const int line);