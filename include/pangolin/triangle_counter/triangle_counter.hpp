/*! \file triangle_counter.hpp
    \brief A Documented file.
    
    Details.
*/

#pragma once

#include <string>

#include "pangolin/config.hpp"

/*! An interface for all triangle counters */
class TriangleCounter
{

public:
  virtual ~TriangleCounter();

public:
  // Triangle-counting phases
  virtual void read_data(const std::string &path) = 0;
  virtual void setup_data();
  virtual size_t count() = 0;

  // available after read_data()
  virtual uint64_t num_edges() = 0; //<! number of edges traversed during triangle counting

  /*! Create a triangle counter

  */
  static TriangleCounter *CreateTriangleCounter(Config &config);
};