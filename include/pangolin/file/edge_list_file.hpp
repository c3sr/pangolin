#pragma once

#include <cassert>
#include <string>

#include "pangolin/edge_list.hpp"

namespace pangolin {

bool endswith(const std::string &base, const std::string &suffix) {
  if (base.size() < suffix.size()) {
    return false;
  }
  return 0 == base.compare(base.size() - suffix.size(), suffix.size(), suffix);
}

class EdgeListFile {

private:
  enum class FileType { TSV, BEL };
  FILE *fp_;
  std::string path_;
  FileType type_;

  template <typename T> size_t read_bel(EdgeTy<T> *ptr, const size_t n) {
    if (fp_ == nullptr) {
      LOG(error, "error reading {} or file was already closed ", path_);
      return 0;
    }
    if (ptr == nullptr) {
      LOG(error, "buffer is a nullptr");
      return 0;
    }
    char *buf = new char[24 * n];
    const size_t numRead = fread(buf, 24, n, fp_);

    // end of file or error
    if (numRead != n) {
      // end of file
      if (feof(fp_)) {
        // do nothing
      }
      // some error
      else if (ferror(fp_)) {
        LOG(error, "Error while reading {}: {}", path_, strerror(errno));
        fclose(fp_);
        fp_ = nullptr;
        assert(0);
      } else {
        LOG(error, "Unexpected error while reading {}", path_);
        assert(0);
      }
    }
    for (size_t i = 0; i < numRead; ++i) {
      uint64_t src, dst;
      std::memcpy(&src, &buf[i * 24 + 8], 8);
      std::memcpy(&dst, &buf[i * 24 + 0], 8);
      ptr[i].first = src;
      ptr[i].second = dst;
      SPDLOG_TRACE(logger::console, "read {} -> {}", ptr[i].first, ptr[i].second);
    }

    // no characters extracted or parsing error
    delete[] buf;
    return numRead;
  }

  template <typename T> size_t read_tsv(EdgeTy<T> *ptr, const size_t n) {
    LOG(critical, "tsv reading unimplemented");
    return 0;
  }

public:
  EdgeListFile(const std::string &path) : path_(path) {
    LOG(debug, "EdgeListFile for \"{}\"", path_);
    if (endswith(path, ".bel")) {
      type_ = FileType::BEL;
    } else if (endswith(path, ".tsv")) {
      type_ = FileType::TSV;
    } else {
      LOG(critical, "no reader for file {}", path);
    }

    fp_ = fopen(path_.c_str(), "r");
    if (nullptr == fp_) {
      LOG(error, "unable to open \"{}\"", path_);
    }
  }

  ~EdgeListFile() {
    if (fp_) {
      fclose(fp_);
      fp_ = nullptr;
    }
  }

  template <typename T> size_t get_edges(std::vector<EdgeTy<T>> &edges, const size_t n) {
    SPDLOG_TRACE(logger::console, "requested {} edges", n);
    edges.resize(n);
    switch (type_) {
    case FileType::BEL: {
      size_t numRead = read_bel(edges.data(), n);
      edges.resize(numRead);
      return numRead;
    }
    case FileType::TSV: {
      size_t numRead = read_tsv(edges.data(), n);
      edges.resize(numRead);
      return numRead;
    }
    default: {
      LOG(critical, "unexpected file type");
      edges.resize(0);
      return 0;
    }
    }
  }
};

} // namespace pangolin