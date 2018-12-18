#pragma once

#include <string>

#include "graph/config.hpp"

class TriangleCounter
{

public:
  virtual ~TriangleCounter();

  // Triangle-counting phases
  virtual void read_data(const std::string &path) = 0;
  virtual void setup_data();
  virtual size_t count() = 0;

  // available after read_data()
  virtual size_t num_edges() = 0;

  static TriangleCounter *CreateTriangleCounter(Config &config);
};