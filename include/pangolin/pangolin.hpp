#pragma once

#include "configure.hpp"
#include "dense/vector.hu"
#include "edge_list.hpp"
#include "generator/complete.hpp"
#include "logger.hpp"
#include "reader/bel_reader.hpp"
#include "reader/gc_tsv_reader.hpp"
#include "sparse/coo.hpp"
#include "sparse/gpu_csr.hpp"
#include "triangle_counter/cpu_triangle_counter.hpp"
#include "triangle_counter/nvgraph_triangle_counter.hpp"
#include "types.hpp"
#include "utilities.hpp"

namespace pangolin {

enum class GraphFormat { CsrCoo };

struct GraphDescription {
  GraphFormat format_;
  int indexSize_;
};

void triangleCount(uint64_t *result, void *graph, const GraphDescription graphDescr);
} // namespace pangolin
