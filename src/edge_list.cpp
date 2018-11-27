
#include <fstream>
#include <limits>

#include "graph/edge_list.hpp"
#include "graph/logger.hpp"

EdgeList EdgeList::read_tsv(const std::string &path)
{
    EdgeList l;

    int64_t src64, dst64, weight64;


    std::ifstream ss(path);

    if (!ss.good())
    {
        LOG(critical, "couldn't open {}", path);
        exit(-1);
    }

    while (ss.good())
    {

        ss >> dst64;
        ss >> src64;
        ss >> weight64;

        if (src64 > std::numeric_limits<Int>::max()) {
            LOG(critical, "{} is too large for sizeof(Int)={}", src64, sizeof(Int));
            exit(-1);
        }
        if (dst64 > std::numeric_limits<Int>::max()) {
            LOG(critical, "{} is too large for sizeof(Int)={}", dst64, sizeof(Int));
            exit(-1);
        }
        Int src = src64;
        Int dst = dst64;

        l.push_back(Edge(src, dst));
    }

    LOG(debug, "EdgeList with {} entries", l.size());
    LOG(debug, "edge 0: {} -> {}", l.begin()->src_, l.begin()->dst_);
    LOG(debug, "last edge: {} -> {}", (l.end()-1)->src_, (l.end()-1)->dst_);

    ss.close();
    return l;
}