#pragma once

#include <cstdint>

#ifdef USE_INT64
typedef int64_t Int;
#else
typedef int32_t Int;
#endif