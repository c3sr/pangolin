#pragma once

#include <cstdlib>
#include <vector>

#include "pangolin/types.hpp"

template <typename INDEX>
class CSR
{
public:
  typedef INDEX index_type;
  virtual const index_type *row_offsets() const = 0;
  virtual const index_type *cols() const = 0;
};
