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

namespace pangolin
{
EdgeListReader *EdgeListReader::from_file(const std::string &path)
{
    if (endswith(path, ".bel"))
    {
        LOG(debug, "creating BELReader");
        return new BELReader(path);
    }
    else if (endswith(path, ".tsv"))
    {
        LOG(debug, "creating GraphChallengeTSVReader");
        return new GraphChallengeTSVReader(path);
    }
    else
    {
        LOG(critical, "Unknown reader for file \"{}\"", path);
        exit(-1);
    }
}
} // namespace pangolin