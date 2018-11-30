#pragma once

#include <cstdint>

#ifdef USE_INT64
typedef int64_t Int;
typedef uint64_t Uint;
#else
typedef int32_t Int;
typedef uint32_t Uint;
#endif