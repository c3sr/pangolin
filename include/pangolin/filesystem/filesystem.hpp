#pragma once

#include <fstream>

namespace pangolin {

namespace filesystem {
/* return true if the path p is a file
 */
bool is_file(const std::string &p) { return std::ifstream(p.c_str()).good(); }

} // namespace filesystem

}; // namespace pangolin