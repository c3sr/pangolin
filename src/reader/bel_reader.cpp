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
        fp_ = nullptr;
    }
}

EdgeListReader *BELReader::clone()
{
    // create a new reader
    auto *reader = new BELReader(path_);

    // match position in fp_
    if (fp_)
    {
        long int tell = ftell(fp_);
        fseek(reader->fp_, tell, SEEK_SET);
    }

    return reader;
}

size_t BELReader::read(Edge *ptr, const size_t num)
{
    assert(fp_ != nullptr);
    assert(ptr != nullptr);
    char *buf = new char[num * 24];
    const size_t numRead = fread(buf, 24, num, fp_);

    // end of file or error
    if (numRead != num)
    {
        // end of file
        if (feof(fp_))
        {
            // do nothing
        }
        // some error
        else if (ferror(fp_))
        {
            LOG(error, "Error while reading {}: {}", path_, strerror(errno));
            fclose(fp_);
            fp_ = nullptr;
            assert(0);
        }
        else
        {
            LOG(error, "Unexpected error while reading {}", path_);
            assert(0);
        }
    }
    for (size_t i = 0; i < numRead; ++i)
    {
        std::memcpy(&ptr[i].first, &buf[i * 24 + 8], 8);
        std::memcpy(&ptr[i].second, &buf[i * 24 + 0], 8);
    }

    // for (size_t i = 0; i < numRead; ++i)
    // {
    //     LOG(debug, "{} {}", ptr[i].first, ptr[i].second);
    // }
    // exit(0);

    // no characters extracted or parsing error
    delete[] buf;
    return numRead;
}

} // namespace graph