#include <fstream>
#include <limits>
#include <sstream>

#include "graph/edge_list.hpp"
#include "graph/reader/gc_tsv_reader.hpp"
#include "graph/logger.hpp"

// read stream until end
static EdgeList read_stream(std::istream &is, std::istream::streampos end)
{
    EdgeList l;
    const Int intMax = std::numeric_limits<Int>::max();

    LOG(debug, "reading from {} until {}", is.tellg(), end);

    for (std::string line; std::getline(is, line);)
    {

        // only check position if we're not reading the whole file
        if (end != -1)
        {
            // if we read past the end for this line, don't record edge
            if (is.tellg() > end)
            {
                LOG(debug, "read past requested end {}", end);
                break;
            }
        }

        std::istringstream iss(line);

        int64_t src64, dst64;
        iss >> dst64;
        iss >> src64;
        // no characters extracted or parsing error
        if (iss.fail())
        {
            break;
        }

        if (src64 > intMax)
        {
            LOG(critical, "{} is too large for sizeof(Int)={}", src64, sizeof(Int));
            exit(-1);
        }
        if (dst64 > intMax)
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
        LOG(debug, "first edge {} -> {}", l.begin()->first, l.begin()->second);
        LOG(debug, "2nd last edge {} -> {}", (l.end() - 2)->first, (l.end() - 2)->second);
        LOG(debug, "last edge {} -> {}", (l.end() - 1)->first, (l.end() - 1)->second);
    }
    return l;
}

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

    fs.clear(); // clear fail and eof bits
    fs.seekg(edgeStart);
    EdgeList l = read_stream(fs, edgeEnd);
    return l;
}

EdgeList GraphChallengeTSVReader::read_edges()
{
    return read_edges(0, -1);
}
