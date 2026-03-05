#pragma once

#include <cstdint>
#include <random>

namespace external::rng {

inline auto& get_thread_local_rng() {
    static thread_local std::mt19937_64 s_rng(42);
    return s_rng;
}

inline void reset_thread_local_rng() {
    get_thread_local_rng().seed(42);
}

} // namespace external::rng
