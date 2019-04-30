#include "pangolin/reader/edge_list_reader.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/reader/bel_reader.hpp"
#include "pangolin/reader/gc_tsv_reader.hpp"

static bool endswith(const std::string &base, const std::string &suffix) {
  if (base.size() < suffix.size()) {
    return false;
  }
  return 0 == base.compare(base.size() - suffix.size(), suffix.size(), suffix);
}

namespace pangolin {

EdgeListReader *EdgeListReader::from_file(const std::string &path) {
  if (endswith(path, ".bel")) {
    LOG(debug, "creating BELReader");
    return new BELReader(path);
  } else if (endswith(path, ".tsv")) {
    LOG(debug, "creating GraphChallengeTSVReader");
    return new GraphChallengeTSVReader(path);
  } else {
    LOG(critical, "Unknown reader for file \"{}\"", path);
    exit(-1);
  }
}

// read all edges from the file
EdgeList EdgeListReader::read_all() {
  const size_t bufSize = 10;
  EdgeList edgeList, buf(bufSize);
  while (true) {
    const size_t numRead = read(buf.data(), 10);
    if (0 == numRead) {
      break;
    }
    edgeList.insert(edgeList.end(), buf.begin(), buf.begin() + numRead);
  }
  return edgeList;
}

} // namespace pangolin