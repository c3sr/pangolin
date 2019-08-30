#pragma once

#include <fstream>

#include "pangolin/edge.hpp"
#include "pangolin/logger.hpp"

namespace pangolin {
/*! Open binary mtx exchange coordinate real general files
 */
template <typename istream> class BmtxStream {

public:
  typedef WeightedDiEdge<int64_t, double> edge_type;

  /*! construct a Bmtx stream by taking ownership of another stream
   */
  BmtxStream(istream &&stream) : stream_(std::move(stream)) { setSizeFromBanner(); }

protected:
  istream stream_;
  uint64_t rows_;
  uint64_t cols_;
  uint64_t entries_;
  char buf_[24];

  void setSizeFromBanner() {
    stream_.seekg(0);
    stream_.read(buf_, 24);
    if (stream_.fail()) {
      LOG(error, "error reading stream");
    }
    rows_ = *reinterpret_cast<int64_t *>(buf_);
    cols_ = *reinterpret_cast<int64_t *>(buf_ + sizeof(int64_t));
    entries_ = *reinterpret_cast<int64_t *>(buf_ + 2 * sizeof(int64_t));
  }

public:
  uint64_t num_rows() const noexcept { return rows_; }
  uint64_t num_cols() const noexcept { return cols_; }
  uint64_t nnz() const noexcept { return entries_; }

  bool readEdge(edge_type &edge) {
    // seek past the header
    if (stream_.tellg() < 24) {
      stream_.seekg(24);
    }

    if (stream_.eof()) {
      SPDLOG_TRACE(logger::console(), "reached eof");
      return false;
    } else {
      SPDLOG_TRACE(logger::console(), "position {} before read", stream_.tellg());
      stream_.read(buf_, 24);
      if (stream_.eof()) {
        return false;
      } else if (stream_.fail()) {
        LOG(error, "error reading stream");
        return false;
      }

      int64_t src = *reinterpret_cast<int64_t *>(buf_);
      int64_t dst = *reinterpret_cast<int64_t *>(buf_ + sizeof(int64_t));
      double weight = *reinterpret_cast<int64_t *>(buf_ + 2 * sizeof(int64_t));
      if (src > int64_t(rows_)) {
        LOG(warn, "{} is greater than expected rows {}", src, rows_);
      }
      if (dst > int64_t(cols_)) {
        LOG(warn, "{} is greater than expected cols {}", dst, cols_);
      }
      edge.src = src - 1;
      edge.dst = dst - 1;
      edge.val = weight;
      return true;
    }
  }
};

/*! Open a new Bmtx stream from path
 */
BmtxStream<std::ifstream> open_bmtx_stream(const std::string &path) {
  auto stream = std::ifstream(path, std::ifstream::binary);
  return BmtxStream<std::ifstream>(std::move(stream));
}

}; // namespace pangolin