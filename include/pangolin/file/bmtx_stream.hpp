#pragma once

#include <fstream>

#include "pangolin/edge.hpp"
#include "pangolin/logger.hpp"

namespace pangolin {
/*! Open binary-coded mtx exchange coordinate-format files
 */
template <typename NodeTy, typename istream> class BmtxStream {

public:
  typedef EdgeTy<NodeTy> Edge;

  /*! construct a Bmtx stream by taking ownership of another stream
   */
  BmtxStream(istream &&stream) : stream_(std::move(stream)) { setSizeFromHeader(); }

protected:
  istream stream_;
  uint64_t rows_;
  uint64_t cols_;
  uint64_t entries_;

  void setSizeFromHeader() {
    int64_t buf[3];
    stream_.seekg(0);
    stream_.read((char *)buf, 24);
    if (stream_.fail()) {
      LOG(error, "error reading stream");
    }
    rows_ = buf[0];
    cols_ = buf[1];
    entries_ = buf[2];
  }

public:
  uint64_t num_rows() const noexcept { return rows_; }
  uint64_t num_cols() const noexcept { return cols_; }
  uint64_t nnz() const noexcept { return entries_; }

  bool readEdge(Edge &edge) {
    // seek past the header
    if (stream_.tellg() < 24) {
      stream_.seekg(24);
    }

    int64_t srcdst[2];
    double weight;

    if (stream_.eof()) {
      return false;
    } else {
      stream_.read((char *)srcdst, 16);
      if (stream_.fail()) {
        LOG(error, "error reading stream");
      }
      stream_.read((char *)&weight, 8);
      if (stream_.fail()) {
        LOG(error, "error reading stream");
      }
      int64_t src = srcdst[0];
      int64_t dst = srcdst[1];
      if (src > rows_) {
        LOG(warn, "{} is greater than expected rows {}", src, rows_);
      }
      if (dst > cols_) {
        LOG(warn, "{} is greater than expected cols {}", dst, cols_);
      }
      edge.first = src;
      edge.second = dst;
      return true;
    }
  }
};

/*! Open a new Bmtx stream from path
 */
template <typename T> BmtxStream<T, std::ifstream> openBmtxStream(const std::string &path) {
  auto stream = std::ifstream(path, std::ifstream::binary);
  return BmtxStream<T, std::ifstream>(std::move(stream));
}

}; // namespace pangolin