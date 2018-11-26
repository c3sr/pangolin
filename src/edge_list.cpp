
#include <fstream>

#include "graph/edge_list.hpp"
#include "graph/logger.hpp"

EdgeList EdgeList::read_tsv(const std::string &path)
{
    EdgeList l;

    Int src, dst, weight;
    std::ifstream ss(path);

    if (!ss.good())
    {
        LOG(critical, "couldn't open {}", path);
        exit(-1);
    }

    std::vector<std::pair<int, long long int>> temp_row_ptrs_vec;

    while (ss.good())
    {
        ss >> dst;
        ss >> src;
        ss >> weight;
        l.push_back(Edge(src, dst));
    }

    LOG(debug, "EdgeList with {} entries", l.size());

    ss.close();
    return l;
}