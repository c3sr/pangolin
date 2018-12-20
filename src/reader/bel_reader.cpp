#include <limits>
#include <sstream>

#include "graph/edge_list.hpp"
#include "graph/reader/bel_reader.hpp"
#include "graph/logger.hpp"

BELReader::BELReader(const std::string &path) : path_(path), is_(path) {}

BELIterator BELReader::begin()
{
    // is_ may have been read before
    is_.clear();                 // reset error state
    is_.seekg(0, std::ios::beg); // go back to beginning
    return BELIterator(is_);
}
BELIterator BELReader::end()
{
    return BELIterator();
}
