#include "pangolin/reader/edge_list_reader.hpp"
#include "pangolin/reader/gc_tsv_reader.hpp"
#include "pangolin/reader/bel_reader.hpp"
#include "pangolin/logger.hpp"

static bool endswith(const std::string &base, const std::string &suffix)
{
    if (base.size() < suffix.size())
    {
        return false;
    }
    return 0 == base.compare(base.size() - suffix.size(), suffix.size(), suffix);
}

PANGOLIN_BEGIN_NAMESPACE()

EdgeListReader *EdgeListReader::from_file(const std::string &path)
{
    if (endswith(path, ".bel"))
    {
        SPDLOG_DEBUG(logger::console, "creating BELReader");
        return new BELReader(path);
    }
    else if (endswith(path, ".tsv"))
    {
        SPDLOG_DEBUG(logger::console, "creating GraphChallengeTSVReader");
        return new GraphChallengeTSVReader(path);
    }
    else
    {
        LOG(critical , "Unknown reader for file \"{}\"", path);
        exit(-1);
    }
}


  // read all edges from the file
EdgeList EdgeListReader::read_all()
{
const size_t bufSize = 10;
EdgeList edgeList, buf(bufSize);
while (true)
{
    const size_t numRead = read(buf.data(), 10);
    if (0 == numRead)
    {
    break;
    }
    edgeList.insert(edgeList.end(), buf.begin(), buf.begin() + numRead);
}
return edgeList;
}


PANGOLIN_END_NAMESPACE()