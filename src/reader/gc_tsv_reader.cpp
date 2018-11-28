#include <fstream>
#include <limits>

#include "graph/edge_list.hpp"
#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/logger.hpp"

GraphChallengeTSVReader::GraphChallengeTSVReader(const std::string &path) : path_(path) {}

// return the position of the beginning of the next line, or the end of the file
static std::istream::streampos next_line_or_eof(const std::string &path, std::istream::streampos start)
{

    std::ifstream newlineFinder(path);
    newlineFinder.seekg(start);

    // read until newline
    newlineFinder.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    if (newlineFinder.eof())
    {
        LOG(trace, "reached EOF in {} after {} while searching for newline", path, start);
        std::ifstream endFinder(path);
        endFinder.seekg(0, endFinder.end);
        return endFinder.tellg();
    }
    else
    {
        LOG(trace, "found newline in {} at {} after {}", path, newlineFinder.tellg(), start);
        return newlineFinder.tellg();
    }
}

long GraphChallengeTSVReader::size()
{
    struct stat stat_buf;
    int rc = stat(path_.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

EdgeList GraphChallengeTSVReader::read_edges(size_t start, size_t end)
{
    std::ifstream fs(path_);

    if (!fs.good())
    {
        LOG(critical, "unable to open {}", path_);
        exit(-1);
    }

    const long sz = size();
    if (sz == -1)
    {
        LOG(critical, "unable to get size for {}", path_);
        exit(-1);
    }
    LOG(trace, "file size is {}", sz);

    if (start == end)
    {
        return EdgeList();
    }

    // find where the edge after start actually starts
    size_t edgeStart;
    if (start == 0)
    {
        edgeStart = 0;
    }
    else
    {
        edgeStart = next_line_or_eof(path_, start);
    }
    LOG(trace, "found edge start after {} at {}", start, edgeStart);

    // find the end of the edge after end
    size_t edgeEnd;
    if (end >= size_t(sz))
    {
        edgeEnd = -1;
    }
    else
    {
        edgeEnd = next_line_or_eof(path_, end);
    }
    LOG(trace, "found edge end after {} at {}", end, edgeEnd);

    fs = std::ifstream(path_);
    fs.seekg(edgeStart);
    EdgeList l = EdgeList::read_tsv(fs, edgeEnd);
    return l;
}