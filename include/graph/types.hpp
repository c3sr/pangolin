#pragma once

#include <cstdint>

#ifdef USE_INT64
typedef int64_t Int;
typedef unsigned long long int Uint;
#else
typedef int32_t Int;
typedef unsigned int Uint;
#endif

static_assert(sizeof(Uint) == sizeof(Int), "expecting Uint to be same size as Int");