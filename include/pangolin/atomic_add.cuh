#pragma once

__device__ inline uint64_t atomicAdd(uint64_t *p, uint64_t val) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long), "expected uint64_t to be unsigned long long");
  return static_cast<unsigned long long>(
      atomicAdd(reinterpret_cast<unsigned long long *>(p), static_cast<unsigned long long>(val)));
}