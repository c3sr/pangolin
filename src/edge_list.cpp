
#include <fstream>
#include <limits>

#include "graph/edge_list.hpp"
#include "graph/logger.hpp"

EdgeList EdgeList::read_tsv(std::istream &is, std::istream::streampos end)
{
    EdgeList l;

    LOG(debug, "reading from {} until {}", is.tellg(), end);

    while (is.good() && is.tellg() < end)
    {
        int64_t src64, dst64, weight64;
        // LOG(debug, "{}", is.tellg());
        is >> dst64;

        // if we fail in the middle of what should be a good edge, the file ends with an empty line
        if (!is.good())
        {
            break;
        }

        // LOG(debug, "{} after {}", is.tellg(), dst64);
        is >> src64;
        // LOG(debug, "{} after {}", is.tellg(), src64);
        is >> weight64;
        // LOG(debug, "{},{} after {}", is.tellg(), is.good(), weight64);

        // If we read past the limit during the reading of this edge, don't record this edge
        if (is.tellg() >= end)
        {
            break;
        }

        if (src64 > std::numeric_limits<Int>::max())
        {
            LOG(critical, "{} is too large for sizeof(Int)={}", src64, sizeof(Int));
            exit(-1);
        }
        if (dst64 > std::numeric_limits<Int>::max())
        {
            LOG(critical, "{} is too large for sizeof(Int)={}", dst64, sizeof(Int));
            exit(-1);
        }
        Int src = src64;
        Int dst = dst64;

        l.push_back(Edge(src, dst));
    }
    LOG(debug, "finished reading stream at {}", is.tellg());

    if (l.size())
    {
        LOG(debug, "first edge {} -> {}", l.begin()->src_, l.begin()->dst_);
        LOG(debug, "2nd last  edge {} -> {}", (l.end() - 2)->src_, (l.end() - 2)->dst_);
        LOG(debug, "last  edge {} -> {}", (l.end() - 1)->src_, (l.end() - 1)->dst_);
    }
    return l;
}

EdgeList EdgeList::read_tsv(const std::string &path)
{
    EdgeList l;

    std::ifstream ss(path);

    if (!ss.good())
    {
        LOG(critical, "couldn't open {}", path);
        exit(-1);
    }

    auto end = ss.seekg(0, std::ios_base::end).tellg();
    ss.seekg(0, std::ios_base::beg);
    l = read_tsv(ss, end);

    LOG(debug, "EdgeList with {} entries", l.size());
    LOG(debug, "edge 0: {} -> {}", l.begin()->src_, l.begin()->dst_);
    LOG(debug, "last edge: {} -> {}", (l.end() - 1)->src_, (l.end() - 1)->dst_);

    ss.close();
    return l;
}