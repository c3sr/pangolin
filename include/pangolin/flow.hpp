#pragma once

namespace pangolin {

enum class AccessKind { OnceExclusive, OnceShared, ManyExclusive, ManyShared, Unknown };

bool is_many(const AccessKind &k) { return k == AccessKind::ManyExclusive || k == AccessKind::ManyShared; }
bool is_once(const AccessKind &k) { return k == AccessKind::OnceExclusive || k == AccessKind::OnceShared; }
bool is_exclusive(const AccessKind &k) { return k == AccessKind::ManyExclusive || k == AccessKind::OnceExclusive; }
bool is_shared(const AccessKind &k) { return k == AccessKind::OnceShared || k == AccessKind::ManyShared; }

class Component {
private:
  enum class Type { CPU, GPU };

public:
  Type type_;
  int id_;
  AccessKind accessKind_;

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
  FlowVector() : stream_(nullptr) { CUDA_RUNTIME(cudaStreamCreate(stream_)); }
  ~FlowVector() {
    if (stream_) {
      CUDA_RUNTIME(cudaStreamDestroy(stream_));
    }
  }

  FlowVector<T> &with_producer(const Component &c) {
    assert(numProducers_ < MAX_COMPONENTS);
    producers[numProducers_++] = c;
    return *this;
  }
  FlowVector<T> with_consumer(const Component &c) {
    assert(numConsumers_ < MAX_COMPONENTS);
    consumers[numConsumers_++] = c;
  }
  void to_producer_async() {}
  void to_consumer_async() {}
  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  /*! get a const view of the container
   */
  void view() {}
};

} // namespace pangolin
