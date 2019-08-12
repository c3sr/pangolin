#pragma once

#include "pangolin/algorithm/zero.cuh"

namespace pangolin {
class DeviceBitVector {
public:
  typedef uint32_t field_type;
  static constexpr size_t BITS_PER_FIELD = sizeof(field_type) * CHAR_BIT; //!< the number of bits stored in each field

private:
  field_type *fields_; //!< the data used to store the bitvector
  size_t size_;        //!< the number of fields in the bitvector
  size_t offset_;      //!< the idx represented by the zeroth bit

public:
  /*! construct a bit vector out of n fields

   */
  __device__ __forceinline__ DeviceBitVector(field_type *fields, size_t n, size_t offset)
      : fields_(fields), size_(n), offset_(offset) {}

  __device__ __forceinline__ DeviceBitVector(field_type *fields, size_t n) : DeviceBitVector(fields, n, 0) {}

  /*! set idx to 1 in the bitvector
   */
  __device__ __forceinline__ void atomic_set(size_t idx) {
    idx -= offset_;
    field_type bits = field_type(1) << (idx % BITS_PER_FIELD);
    idx = idx / BITS_PER_FIELD;
    atomicOr(&fields_[idx], bits);
  }

  /*! true if the bit is 1, 0 otherwise
   */
  __device__ __forceinline__ bool get(size_t idx) const {
    idx -= offset_;
    field_type bits = fields_[idx / BITS_PER_FIELD];
    return field_type(1) & (bits >> (idx % BITS_PER_FIELD));
  }

  /*! block-collaborative zero of bits corresponding to [lowIdx, highIdx]
   */
  __device__ __forceinline__ void block_clear_inclusive(size_t lowIdx, size_t highIdx) {
    size_t lowIdxOff = lowIdx - offset_;
    size_t highIdxOff = highIdx - offset_;

    // the fields that lowIdx and highIdx appear in
    size_t lowFieldIdx = lowIdxOff / BITS_PER_FIELD;
    size_t highFieldIdx = highIdxOff / BITS_PER_FIELD;

    // clear low-high inclusive
    block_zero(&fields_[lowFieldIdx], highFieldIdx - lowFieldIdx + 1);
  }

  /*! block-collaborative zero of bits corresponding to [lowIdx, highIdx]
   */
  __device__ __forceinline__ void warp_clear_inclusive(size_t lowIdx, size_t highIdx) {
    size_t lowIdxOff = lowIdx - offset_;
    size_t highIdxOff = highIdx - offset_;

    // the fields that lowIdx and highIdx appear in
    size_t lowFieldIdx = lowIdxOff / BITS_PER_FIELD;
    size_t highFieldIdx = highIdxOff / BITS_PER_FIELD;

    // clear low-high inclusive
    warp_zero(&fields_[lowFieldIdx], highFieldIdx - lowFieldIdx + 1);
  }

  /*! the number of bits in the bitvector
   */
  __device__ __forceinline__ size_t size() const { return size_ * BITS_PER_FIELD; }

  /*! one larger than the maximum index settable in the bit vector
   */
  __device__ __forceinline__ size_t end_idx() const { return offset_ + size_ * BITS_PER_FIELD; }

  /*! return a raw pointer to the data
   */
  __device__ __forceinline__ field_type *data() const { return fields_; }

  /*! return a raw pointer to the data
   */
  __device__ __forceinline__ void set_offset(size_t offset) { offset_ = offset; }
};

} // namespace pangolin