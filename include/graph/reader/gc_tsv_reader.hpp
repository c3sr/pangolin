#pragma once

/*
Reader for GraphChallenge format TSV files.
Each line should be ASCII tab-separated dst, src, weight,
sorted with increasing src, and within each src increasing dst
*/

#include <string>

#include "graph/edge_list.hpp"

class GraphChallengeTSVReader
{

private:
  std::string path_;

public:
  GraphChallengeTSVReader(const std::string &path);

  // read_edges(0, N/2)
  // read_edges(N/2, N)
  /*
    Read edges starting from the first complete edge after start to the
    last complete edge after end. So, a Reader on a file with length N
    would be completely covered like this:
    read_edges(0, N/2)
    read_edges(N/2, N)
    If start is 0, start there
    if end is past EOF, read until EOF
    */
  EdgeList read_edges(size_t start, size_t end);

  // Read all edges
  EdgeList read_edges();

  long size();
};