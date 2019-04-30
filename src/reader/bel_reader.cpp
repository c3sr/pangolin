#include <cstdio>
#include <limits>

#include "pangolin/edge_list.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/reader/bel_reader.hpp"

namespace pangolin {

BELReader::BELReader(const std::string &path) : fp_(nullptr), path_(path) { fp_ = fopen(path.c_str(), "r"); }

BELReader::~BELReader() {
  if (fp_) {
    fclose(fp_);
    fp_ = nullptr;
  }
}

EdgeListReader *BELReader::clone() {
  // create a new reader
  auto *reader = new BELReader(path_);

  // match position in fp_
  if (fp_) {
    long int tell = ftell(fp_);
    fseek(reader->fp_, tell, SEEK_SET);
  }

  return reader;
}

/*! read edges into a buffer

\param ptr pointer to at least num * sizeof(Edge) bytes
\param num number of edges to read
\returns number of edges read
*/
size_t BELReader::read(Edge *ptr, const size_t num) {
  if (fp_ == nullptr) {
    LOG(error, "error reading {} or file was already closed ", path_);
    return 0;
  }
  if (ptr == nullptr) {
    LOG(error, "buffer is a nullptr");
    return 0;
  }
  char *buf = new char[num * 24];
  const size_t numRead = fread(buf, 24, num, fp_);

  // end of file or error
  if (numRead != num) {
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
    std::memcpy(&ptr[i].first, &buf[i * 24 + 8], 8);
    std::memcpy(&ptr[i].second, &buf[i * 24 + 0], 8);
  }

  // for (size_t i = 0; i < numRead; ++i)
  // {
  //     LOG(debug, "{} {}", ptr[i].first, ptr[i].second);
  // }
  // exit(0);

  // no characters extracted or parsing error
  delete[] buf;
  return numRead;
}

} // namespace pangolin