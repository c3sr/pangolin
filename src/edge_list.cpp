
#include <fstream>

#include "graph/edge_list.hpp"
#include "graph/logger.hpp"

EdgeList EdgeList::read_tsv(const std::string &path)
{
    EdgeList l;

    Int src, dst, weight;
    std::ifstream ss(path);
    std::vector<std::pair<int, long long int>> temp_row_ptrs_vec;

    Int edgecount = 0;
    Int nodecount = 0;
    Int prevkey = -1;

    while (ss.good())
    {
        ss >> src;
        ss >> dst;
        ss >> weight;
        l.push_back(Edge(src, dst));
    }

    LOG(debug, "EdgeList with {} entries", l.size());

    ss.close();
    return l;
}