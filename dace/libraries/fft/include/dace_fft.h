// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace dace {

namespace fft {

// Vendor planners are not thread-safe; plans are made under this one lock.
static inline std::mutex& PlannerMutex() {
  static std::mutex mutex;
  return mutex;
}

// Vendor plans keyed by everything a plan depends on: made once, destroyed when the program unloads.
template <typename Plan>
class PlanCache {
 public:
  using Destroy = void (*)(Plan);

  explicit PlanCache(Destroy destroy) : destroy_(destroy) {}
  PlanCache(PlanCache const&) = delete;
  PlanCache& operator=(PlanCache const&) = delete;

  ~PlanCache() {
    for (std::pair<const std::vector<long long>, Plan>& entry : plans_) destroy_(entry.second);
  }

  // The plan for `key`, made by `make()` on first use.
  template <typename Make>
  Plan Get(std::vector<long long> const& key, Make make) {
    std::lock_guard<std::mutex> lock(PlannerMutex());
    typename std::map<std::vector<long long>, Plan>::iterator found = plans_.find(key);
    if (found != plans_.end()) return found->second;
    Plan plan = make();
    plans_.emplace(key, plan);
    return plan;
  }

 private:
  Destroy destroy_;
  std::map<std::vector<long long>, Plan> plans_;
};

namespace {

// This program's cache of `Plan`s. Internal linkage: a function-template static with external linkage is one object
// per process, shared by every loaded program, and two programs may bind the same planner symbols to different
// libraries (FFTW3, or MKL's FFTW3 interface once MKL is loaded), so a plan one made would reach the other.
template <typename Plan>
PlanCache<Plan>& Plans(typename PlanCache<Plan>::Destroy destroy) {
  static PlanCache<Plan> cache(destroy);
  return cache;
}

}  // namespace

}  // namespace fft

}  // namespace dace
