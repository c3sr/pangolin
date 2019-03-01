
#include "pangolin/algorithm/triangle_count.cuh"
#include "pangolin/pangolin.hpp"

namespace pangolin {

void triangleCount(uint64_t *result, void *graph, const GraphDescription graphDescr) {
  if (nullptr == result) {
    LOG(critical, "result pointer is null");
  }
  if (nullptr == graph) {
    LOG(critical, "graph pointer is null");
  }

  switch (graphDescr.format_) {
  case GraphFormat::CsrCoo: {
    if (graphDescr.indexSize_ == 8) {
      LOG(error, "unhandled index size {}", graphDescr.indexSize_);
    } else if (graphDescr.indexSize_ == 4) {
      auto g = reinterpret_cast<COO<int> *>(graph);
      *result = triangle_count(*g);
      return;
    } else {
      LOG(error, "unhandled index size {}", graphDescr.indexSize_);
      return;
    }
    break;
  }
  }
}

} // namespace pangolin