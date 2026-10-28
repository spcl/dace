// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <map>
#include <mutex>
#include <vector>

namespace dace {

namespace fft {

// Vendor plans keyed by everything a plan depends on. A program-state field: made once per program, destroyed with it.
template <typename Plan>
class PlanCache {
 public:
  using Destroy = void (*)(Plan);

  PlanCache() = default;
  PlanCache(PlanCache const&) = delete;
  PlanCache& operator=(PlanCache const&) = delete;

  ~PlanCache() {
    for (auto& entry : plans_) destroy_(entry.second);
  }

  // The plan for `key`, made by `make()` on first use. `destroy` is taken here, not at construction, so a program
  // links only the planner of the precision it uses. Vendor planners are not thread-safe: plans are made under a lock.
  template <typename Make>
  Plan Get(std::vector<long long> const& key, Destroy destroy, Make make) {
    std::lock_guard<std::mutex> lock(mutex_);
    destroy_ = destroy;
    auto found = plans_.find(key);
    if (found != plans_.end()) return found->second;
    Plan plan = make();
    plans_.emplace(key, plan);
    return plan;
  }

 private:
  std::mutex mutex_;
  Destroy destroy_ = nullptr;
  std::map<std::vector<long long>, Plan> plans_;
};

}  // namespace fft

}  // namespace dace
