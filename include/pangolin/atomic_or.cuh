#pragma once

/*! \brief Convenience wrapper for atomicOr(uint64_t*, uint64_t)
 */
__device__ inline uint64_t atomicOr(uint64_t *p, uint64_t val) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long), "expected uint64_t to be unsigned long long");
  return static_cast<unsigned long long>(
      atomicOr(reinterpret_cast<unsigned long long int *>(p), static_cast<unsigned long long int>(val)));
}

/*! \brief Convenience wrapper for atomicAdd(volatile uint64_t*, uint64_t)
 */
__device__ inline uint64_t atomicOr(volatile uint64_t *p, uint64_t val) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long), "expected uint64_t to be unsigned long long");
  volatile unsigned long long *vull = reinterpret_cast<volatile unsigned long long *>(p);
  unsigned long long *ull = const_cast<unsigned long long *>(vull);
  return static_cast<unsigned long long>(
      atomicOr(ull, static_cast<unsigned long long>(val)));
}
