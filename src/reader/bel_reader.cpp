#include <limits>
#include <cstdio>

#include "graph/edge_list.hpp"
#include "graph/reader/bel_reader.hpp"
#include "graph/logger.hpp"

namespace graph
{

BELReader::BELReader(const std::string &path) : fp_(nullptr), path_(path)
{
    fp_ = fopen(path.c_str(), "r");
}

BELReader::~BELReader()
{
    if (fp_)
    {
        fclose(fp_);
    }
    fp_ = nullptr;
}

size_t BELReader::read(Edge *ptr, const size_t num)
{
    assert(fp_ != nullptr);
    assert(ptr != nullptr);
    const size_t numRead = fread(ptr, sizeof(Edge), num, fp_);

    // end of file or error
    if (numRead != num)
    {
        // end of file
        if (feof(fp_))
        {
            return 0;
        }
        // some error
        else if (ferror(fp_))
        {
            LOG(error, "Error while reading {}", path_);
            assert(0);
            fclose(fp_);
            fp_ = nullptr;
        }
        else
        {
            LOG(error, "Unexpected error while reading {}", path_);
            assert(0);
        }
    }

    // no characters extracted or parsing error
    return numRead;
}

} // namespace graph