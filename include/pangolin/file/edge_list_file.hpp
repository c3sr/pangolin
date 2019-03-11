#pragma once

#include <cassert>
#include <string>

#include "pangolin/edge_list.hpp"

namespace pangolin {

/*! check if base string ends with suffix string

/returns true if base ends with suffix, false otherwise
*/
bool endswith(const std::string &base,  //!< [in] the base string
              const std::string &suffix //!< [in] the suffix to check for
) {
  if (base.size() < suffix.size()) {
    return false;
  }
  return 0 == base.compare(base.size() - suffix.size(), suffix.size(), suffix);
}

/*! a class representing an edge list file
 */
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

    assert(ptr != nullptr);
    assert(fp_ != nullptr);

    size_t i = 0;
    for (; i < n; ++i) {
      long long unsigned dst, src, weight;
      const size_t numFilled = fscanf(fp_, "%llu %llu %llu", &dst, &src, &weight);
      if (numFilled != 3) {
        if (feof(fp_)) {
          return i;
        } else if (ferror(fp_)) {
          LOG(error, "Error while reading {}: {}", path_, strerror(errno));
          return i;
        } else {
          LOG(critical, "Unexpected error while reading {}", path_);
          exit(-1);
        }
      }
      ptr[i].first = static_cast<T>(src);
      ptr[i].second = static_cast<T>(dst);
    }
    return i;
  }

public:
  /*! \brief Construct an EdgeListFile

    Supports GraphChallenge TSV or BEL files
  */
  EdgeListFile(const std::string &path //!< [in] the path of the file
               )
      : path_(path) {
    LOG(debug, "EdgeListFile for \"{}\"", path_);
    if (endswith(path, ".bel")) {
      type_ = FileType::BEL;
    } else if (endswith(path, ".tsv")) {
      type_ = FileType::TSV;
    } else {
      LOG(critical, "no reader for file {}", path);
      exit(-1);
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

  /*! \brief attempt to read n edges from the file

    \tparam T the node ID type
    \returns the number of edges read

  */
  template <typename T>
  size_t
  get_edges(std::vector<EdgeTy<T>> &edges, //!< [out] the read edges. Resized to the number of successfully read edges
            const size_t n                 //!< [in] the number of edges to try to read
  ) {
    SPDLOG_TRACE(logger::console, "requested {} edges", n);
    edges.resize(n);

    size_t numRead;
    switch (type_) {
    case FileType::BEL: {
      numRead = read_bel(edges.data(), n);
      break;
    }
    case FileType::TSV: {
      numRead = read_tsv(edges.data(), n);
      break;
    }
    default: {
      LOG(critical, "unexpected file type");
      exit(-1);
    }
    }
    edges.resize(numRead);
    return numRead;
  }
};

} // namespace pangolin