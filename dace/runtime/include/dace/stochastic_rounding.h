// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_SROUND_H
#define __DACE_SROUND_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thread>

namespace dace {
class float32sr {
private:
    float value;

    static constexpr double FLOATMIN_F32 = 1.1754943508222875e-38;

    // Random Number Generation using Linear Congruential Generator
    static inline uint64_t& get_rng_state_64() {
        static thread_local uint64_t rng_state_64 = 1;
        return rng_state_64;
    }

    static inline uint64_t lcg64() {
        uint64_t& rng_state = get_rng_state_64();
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL; // 64-bit LCG constants
        return rng_state;
    }

    static inline uint64_t xorshift64() {
        uint64_t& rng_state = get_rng_state_64();
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return rng_state;
    }

    static inline float stochastic_round(double x) {
        // Stochastic Rounding to a float32 from a double
        // Input:   [ Sign |  Exponent (11) | Mantissa (52) ]   <- double (64 bits)
        // Output:  [ Sign |  Exponent (8)  | Mantissa (23) ]   <- float  (32 bits)
        //
        // double:  S | EEEEEEEEEEE | MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM[29 excess bits]
        //                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^
        //                                                       These 29 bits are lost when converting to float32.
        //
        // float:   S | EEEEEEEE    | MMMMMMMMMMMMMMMMMMMMMMMMMMM
        //                 ^                   ^
        //              8 exp bits    23 mantissa bits (kept)
        //
        // We add random noise to the excess bits then truncate them to ensure the result is a representable float32.
        // If the perturbation is enough, the least significant bit of the rounded value is incremented.
        // This is equivalent to rounding up to the nearest representable number with probability 1 - distance to the
        // next representable number. The modified double is finally cast to float32.

        uint64_t rbits = get_random_u64(); // it's quicker to use u64 over u32, perhaps due casting?

        // if x is subnormal, round randomly; N.B fast-math forces subnormal values to 0
        if (abs(x) < FLOATMIN_F32) {
            return static_cast<float>(x + rand_subnormal(rbits));
        }

        uint64_t bits = double_to_bits(x);
        const uint64_t mask = 0x000000001FFFFFFF;  // mask last 29 surplus bits of the double prec mantissa

        bits += (rbits & mask);  // add stochastic perturbation
        bits &= ~mask;           // truncate lower bits
        double rounded = bits_to_double(bits);
        return static_cast<float>(rounded);
    }

    static inline double rand_subnormal(uint64_t rbits) {
        // Impl of https://github.com/milankl/StochasticRounding.jl/blob/ff86c801ddd15b4182cb551b84811eb24295a48f/src/types.jl#L99C1-L106C4
        int lz = __builtin_clzll(rbits); // Count leading zeros

        uint64_t exponent = static_cast<uint64_t>(872 - lz); // Compute biased exponent
        exponent <<= 52;

        uint64_t sign = (rbits >> 63) << 63; // Take highest bit as sign

        uint64_t mantissa = rbits & ((1ULL << 52) - 1); // Mask to get mantissa (only lower 52 bits)

        uint64_t bits = sign | exponent | mantissa;
        double result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    static inline uint64_t double_to_bits(double x) {
        uint64_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        return bits;
    }

    static inline double bits_to_double(uint64_t bits) {
        double x;
        std::memcpy(&x, &bits, sizeof(x));
        return x;
    }

    static inline uint64_t get_random_u64() {
        // Default stochastic rounding mode: Linear congruential generator (lcg64)
        // Alternative SR modes: Circular buffer (get_preloaded_random_u64), Xorshift64 generator (xorshift64)
        return lcg64();
    }


public:
    float32sr() : value(0.0f) {}
    // Small integers are exactly representable in float32
    float32sr(int8_t v) : value(static_cast<float>(v)) {}
    float32sr(int16_t v) : value(static_cast<float>(v)) {}
    // Larger integers may lose precision, apply stochastic rounding
    float32sr(int32_t v) : value(stochastic_round(static_cast<double>(v))) {}
    float32sr(int64_t v) : value(stochastic_round(static_cast<double>(v))) {}
    float32sr(float v) : value(v) {}
    float32sr(double v) : value(stochastic_round(v)) {}

    operator float() const { return value; }
    operator float*() { return &value; }
    operator const float*() const { return &value; }

    friend std::istream& operator>>(std::istream& is, float32sr& obj);
    friend std::ostream& operator<<(std::ostream& os, const float32sr& obj);

    // Circular buffer for random number generation
    // The idea is a large buffer of rands is created once on init then
    // reused by all threads for rounding.
    static uint64_t get_preloaded_random_u64() {
      struct SharedRandomQueue {
          std::array<uint64_t, 10000> data;
          std::atomic<bool> initialized{false};

          SharedRandomQueue() {
              initialize();
          }

          void initialize() {
              for (auto& x : data) {
                  x = float32sr::get_random_u64();
              }
              initialized.store(true, std::memory_order_release);
          }

          uint64_t get(size_t idx) const {
              return data[idx % data.size()];
          }
      };

      static SharedRandomQueue shared_queue;
      thread_local static size_t thread_index = 0;

      size_t current_index = thread_index;
      thread_index = (thread_index + 1) % shared_queue.data.size();

      return shared_queue.get(current_index);
    }

    #define DEFINE_COMPARISON(op) \
        bool operator op (const float32sr& other) const { \
            return value op other.value; \
        }

    DEFINE_COMPARISON(==)
    DEFINE_COMPARISON(!=)
    DEFINE_COMPARISON(<)
    DEFINE_COMPARISON(<=)
    DEFINE_COMPARISON(>)
    DEFINE_COMPARISON(>=)

    #undef DEFINE_COMPARISON
    #define DEFINE_COMPARISON(op) \
        bool operator op (const float& other) const { \
            return value op other; \
        }

    DEFINE_COMPARISON(==)
    DEFINE_COMPARISON(!=)
    DEFINE_COMPARISON(<)
    DEFINE_COMPARISON(<=)
    DEFINE_COMPARISON(>)
    DEFINE_COMPARISON(>=)

    #undef DEFINE_COMPARISON

    //==================
    // Assignments
    //==================
    float32sr& operator=(const float& v) {
        value = v;
        return *this;
    }

    float32sr& operator=(const double& v) {
        value = stochastic_round(v);
        return *this;
    }

    // Small integers (int8, int16) are exactly representable in float32
    float32sr& operator=(const int8_t& v) {
        value = static_cast<float>(v);
        return *this;
    }

    float32sr& operator=(const int16_t& v) {
        value = static_cast<float>(v);
        return *this;
    }

    // Larger integers may exceed float32's 24-bit precision, apply SR
    float32sr& operator=(const int32_t& v) {
        value = stochastic_round(static_cast<double>(v));
        return *this;
    }

    float32sr& operator=(const int64_t& v) {
        value = stochastic_round(static_cast<double>(v));
        return *this;
    }

    // Primary operators (SR + SR -> SR with stochastic rounding)
    float32sr operator+(const float32sr& other) const {
        return float32sr(stochastic_round(static_cast<double>(value) + static_cast<double>(other.value)));
    }

    float32sr operator*(const float32sr& other) const {
        return float32sr(stochastic_round(static_cast<double>(value) * static_cast<double>(other.value)));
    }

    float32sr operator-(const float32sr& other) const {
        return float32sr(stochastic_round(static_cast<double>(value) - static_cast<double>(other.value)));
    }

    float32sr operator/(const float32sr& other) const {
        return float32sr(stochastic_round(static_cast<double>(value) / static_cast<double>(other.value)));
    }

    // Mixed operators (SR + float -> float, no stochastic rounding)
    float operator+(const float& other) const { return value + other; }
    float operator-(const float& other) const { return value - other; }
    float operator*(const float& other) const { return value * other; }
    float operator/(const float& other) const { return value / other; }

    // Mixed operators (SR + int -> float)
    float operator+(const int& other) const { return value + other; }
    float operator-(const int& other) const { return value - other; }
    float operator*(const int& other) const { return value * other; }
    float operator/(const int& other) const { return value / other; }

    // Compound assignment operators
    float32sr& operator+=(const float32sr& other) {
        value = stochastic_round(static_cast<double>(value) + static_cast<double>(other.value));
        return *this;
    }

    float32sr& operator-=(const float32sr& other) {
        value = stochastic_round(static_cast<double>(value) - static_cast<double>(other.value));
        return *this;
    }

    float32sr& operator*=(const float32sr& other) {
        value = stochastic_round(static_cast<double>(value) * static_cast<double>(other.value));
        return *this;
    }

    float32sr& operator/=(const float32sr& other) {
        value = stochastic_round(static_cast<double>(value) / static_cast<double>(other.value));
        return *this;
    }
};

inline std::istream& operator>>(std::istream& is, float32sr& obj) {
    is >> obj.value;
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const float32sr& obj) {
    os << obj.value;
    return os;
}

}  // namespace dace

#endif  // __DACE_SROUND_H
