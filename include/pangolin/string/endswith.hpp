#pragma once

#include <string>

namespace pangolin {
namespace string {
/*! check if base string ends with suffix string

/returns true if base ends with suffix, false otherwise
*/
bool endswith(const std::string &base,  //!< [in] the base string
              const std::string &suffix //!< [in] the suffix to check for
) {
  if (base.size() < suffix.size()) {
    return false;
  }
  return 0 == base.compare(base.size() - suffix.size(), suffix.size(), suffix);
}
} // namespace string
} // namespace pangolin