#include "graph/reader/edge_list_reader.hpp"

#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/reader/bel_reader.hpp"
#include "graph/logger.hpp"

static bool endswith(const std::string &base, const std::string &suffix)
{
    if (base.size() < suffix.size())
    {
        return false;
    }
    return 0 == base.compare(base.size() - suffix.size(), suffix.size(), suffix);
}

namespace graph
{
EdgeListReader *EdgeListReader::from_file(const std::string &path)
{
    if (endswith(path, ".bel"))
    {
        return new BELReader(path);
    }
    else if (endswith(path, ".tsv"))
    {
        return new GraphChallengeTSVReader(path);
    }
    else
    {
        LOG(critical, "Unknown reader for file \"{}\"", path);
        exit(-1);
    }
}
} // namespace graph