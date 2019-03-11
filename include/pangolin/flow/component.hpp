#pragma once

#include "access_kind.hpp"

namespace pangolin {

class Component {

private:
  enum class Type { CPU, GPU, UNKNOWN };

public:
  Type type_;
  int id_;
  AccessKind accessKind_;
  Component() : type_(Type::UNKNOWN), accessKind_(AccessKind::Unknown) {}
  bool is_cpu() const { return Type::CPU == type_; }
  int id() const { return id_; }

private:
  Component(Type type, int id) : type_(type), id_(id), accessKind_(AccessKind::Unknown) {}
  Component(Type type, int id, AccessKind accessKind) : type_(type), id_(id), accessKind_(accessKind) {}

public:
  static Component CPU(int id) { return Component(Type::CPU, id); }
  static Component CPU(int id, AccessKind accessKind) { return Component(Type::CPU, id, accessKind); }
  static Component GPU(int id, AccessKind accessKind) { return Component(Type::GPU, id, accessKind); }
};

} // namespace pangolin