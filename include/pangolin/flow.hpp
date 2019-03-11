#pragma once

#include "pangolin/utilities.hpp"

namespace pangolin {

enum class AccessKind { OnceExclusive, OnceShared, ManyExclusive, ManyShared, Unknown };

bool is_many(const AccessKind &k) { return k == AccessKind::ManyExclusive || k == AccessKind::ManyShared; }
bool is_once(const AccessKind &k) { return k == AccessKind::OnceExclusive || k == AccessKind::OnceShared; }
bool is_exclusive(const AccessKind &k) { return k == AccessKind::ManyExclusive || k == AccessKind::OnceExclusive; }
bool is_shared(const AccessKind &k) { return k == AccessKind::OnceShared || k == AccessKind::ManyShared; }

class Component {

private:
  enum class Type { CPU, GPU, UNKNOWN };

public:
  Type type_;
  int id_;
  AccessKind accessKind_;
  Component() : type_(Type::UNKNOWN), accessKind_(AccessKind::Unknown) {}

private:
  Component(Type type, int id) : type_(type), id_(id), accessKind_(AccessKind::Unknown) {}
  Component(Type type, int id, AccessKind accessKind) : type_(type), id_(id), accessKind_(accessKind) {}

public:
  static Component CPU(int id) { return Component(Type::CPU, id); }
  static Component CPU(int id, AccessKind accessKind) { return Component(Type::CPU, id, accessKind); }
  static Component GPU(int id, AccessKind accessKind) { return Component(Type::GPU, id, accessKind); }
};

template <typename T, size_t MAX_COMPONENTS = 16> class FlowVector {

private:
  cudaStream_t stream_;
  size_t numConsumers_;
  size_t numProducers_;
  Component producers_[MAX_COMPONENTS];
  Component consumers_[MAX_COMPONENTS];

public:
  FlowVector() : stream_(0), numProducers_(0), numConsumers_(0) { CUDA_RUNTIME(cudaStreamCreate(&stream_)); }
  ~FlowVector() {
    if (stream_) {
      CUDA_RUNTIME(cudaStreamDestroy(stream_));
    }
  }

  FlowVector<T> &add_producer(const Component &c) {
    assert(numProducers_ < MAX_COMPONENTS);
    producers_[numProducers_++] = c;
    return *this;
  }
  FlowVector<T> &add_consumer(const Component &c) {
    assert(numConsumers_ < MAX_COMPONENTS);
    consumers_[numConsumers_++] = c;
    return *this;
  }

  static FlowVector<T> from(std::initializer_list<Component> &cs) {
    FlowVector<T> v;
    for (const auto &c : cs) {
      v.add_producer(c);
    }
    return v;
  }

  static FlowVector<T> to(std::initializer_list<Component> &cs) {
    FlowVector<T> v;
    for (const auto &c : cs) {
      v.add_consumer(c);
    }
    return v;
  }

  void to_producer_async() {}
  void to_producer() {
    to_producer_async();
    sync();
  }

  void to_consumer_async() {}
  void to_consumer() {
    to_consumer_async();
    sync();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
};

} // namespace pangolin
