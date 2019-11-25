#pragma once

#include <cassert>
#include <string>
#include <vector>

#include "pangolin/edge.hpp"
#include "pangolin/logger.hpp"

namespace pangolin {

namespace file {

class TSV {
private:
  FILE *fp_;
  std::string path_;

public:
  typedef WeightedDiEdge<int64_t, int64_t> edge_type;

  TSV(const std::string &path) : path_(path) {
    fp_ = fopen(path_.c_str(), "r");
    if (nullptr == fp_) {
      LOG(error, "unable to open \"{}\"", path_);
    }
  }

  ~TSV() { fclose(fp_); }

  bool read_edge(edge_type &e) {
    long long int src, dst, weight;
    const size_t numFilled = fscanf(fp_, "%lli %lli %lli", &dst, &src, &weight);
    if (numFilled != 3) {
      if (feof(fp_)) {
        return false;
      } else if (ferror(fp_)) {
        LOG(error, "Error while reading {}: {}", path_, strerror(errno));
        exit(-1);
      } else {
        LOG(critical, "Unexpected error while reading {}", path_);
        exit(-1);
      }
    }
    e.src = src;
    e.dst = dst;
    e.val = weight;
    return true;
  }

  std::vector<edge_type> read_edges() {
    edge_type e;
    std::vector<edge_type> result;
    while (read_edge(e)) {
      result.push_back(e);
    }
    return result;
  }
};

} // namespace file

} // namespace pangolin