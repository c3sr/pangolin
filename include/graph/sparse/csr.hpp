#pragma once

#include <cstdlib>
#include <vector>

#include "graph/types.hpp"

template <typename INDEX, typename SCALAR>
class CSR
{
  public:
    typedef INDEX index_type;
    typedef SCALAR scalar_type;
    virtual index_type *row_offsets() = 0;
    virtual scalar_type *data() = 0;
    virtual size_t num_nodes() const = 0;
    virtual size_t num_edges() const = 0;
};
