#pragma once

namespace pangolin {
enum class AccessKind { OnceExclusive, OnceShared, ManyExclusive, ManyShared, Unknown };

bool is_many(const AccessKind &k) { return k == AccessKind::ManyExclusive || k == AccessKind::ManyShared; }
bool is_once(const AccessKind &k) { return k == AccessKind::OnceExclusive || k == AccessKind::OnceShared; }
bool is_exclusive(const AccessKind &k) { return k == AccessKind::ManyExclusive || k == AccessKind::OnceExclusive; }
bool is_shared(const AccessKind &k) { return k == AccessKind::OnceShared || k == AccessKind::ManyShared; }
}; // namespace pangolin