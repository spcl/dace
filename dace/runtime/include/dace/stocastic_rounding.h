#ifndef __DACE_SROUND_H
#define __DACE_SROUND_H

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thread>

#ifdef __CUDACC__
#define DACE_HOST_DEVICE __host__ __device__
#include <curand_kernel.h>
#else
#define DACE_HOST_DEVICE
#endif

namespace dace {
class float32sr {
private:
    float value;

    static constexpr double FLOATMIN_F32 = 1.1754943508222875e-38;

    #if !defined(__CUDA_ARCH__)
    DACE_HOST_DEVICE static inline uint64_t& get_rng_state_64() {
        static thread_local uint64_t rng_state_64 = 1;
        return rng_state_64;
    }

    DACE_HOST_DEVICE static inline uint64_t lcg64() {
        uint64_t& rng_state = get_rng_state_64();
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL; // 64-bit LCG constants
        return rng_state;
    }

    DACE_HOST_DEVICE static inline uint64_t xorshift64() {
        uint64_t& rng_state = get_rng_state_64();
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return rng_state;
    }
    #endif

    DACE_HOST_DEVICE static float stochastic_round(double x) {
        // uint64_t rbits = lcg64();
        uint64_t rbits = get_random_u64(); // it's quicker to use u64 over u32, perhaps due casting?
        // uint64_t rbits = get_preloaded_random_u64(); // "circular buffer"
        // uint64_t rbits = 0; // "constant" rounding
        // uint64_t rbits = xorshift64();

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

    DACE_HOST_DEVICE static double rand_subnormal(uint64_t rbits) {
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


    DACE_HOST_DEVICE static uint64_t double_to_bits(double x) {
    #if defined(__CUDA_ARCH__)
        return __double_as_longlong(x);
    #else
        uint64_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        return bits;
    #endif
    }


    DACE_HOST_DEVICE static double bits_to_double(uint64_t bits) {
    #if defined(__CUDA_ARCH__)
        return __longlong_as_double(bits);
    #else
        double x;
        std::memcpy(&x, &bits, sizeof(x));
        return x;
    #endif
    }


    DACE_HOST_DEVICE static uint64_t get_random_u64() {
    #if defined(__CUDA_ARCH__)
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint64_t seed = clock64() + tid;

        curandStatePhilox4_32_10_t state;
        curand_init(seed, /* subsequence */ tid, /* offset */ 0, &state);

        uint64_t low = curand(&state); // first 32 bits all 0, last 32 bits are populated by curand
        return low;
    #else
        return lcg64();
    #endif
    }

    #if !defined(__CUDA_ARCH__)
    static std::mt19937_64& get_rng_mt() {
        static std::mt19937_64 rng(std::random_device{}());
        return rng;
    }
    #endif

    #if !defined(__CUDA_ARCH__)
    static std::mt19937& get_rng_single() {
        static std::mt19937 rng(std::random_device{}());
    return rng;
}
#endif


public:
    DACE_HOST_DEVICE float32sr() : value(0.0f) {}
    DACE_HOST_DEVICE float32sr(int v) : value(static_cast<double>(v)) {}
    DACE_HOST_DEVICE float32sr(float v) : value(v) {}
    DACE_HOST_DEVICE float32sr(double v) : value(stochastic_round(v)) {}

    DACE_HOST_DEVICE operator float() const { return value; }
    DACE_HOST_DEVICE operator float*() { return &value; }
    DACE_HOST_DEVICE operator const float*() const { return &value; }

    friend std::istream& operator>>(std::istream& is, float32sr& obj);
    friend std::ostream& operator<<(std::ostream& os, const float32sr& obj);


    DACE_HOST_DEVICE static uint64_t get_preloaded_random_u64() {
    #if defined(__CUDA_ARCH__)
        __shared__ uint64_t shared_randoms[1000];
        __shared__ bool initialized;
        __shared__ int index;

        if (threadIdx.x == 0 && !initialized) {
            for (int i = 0; i < 1000; ++i) {
                shared_randoms[i] = get_random_u64();
            }
            index = 0;
            initialized = true;
        }

        __syncthreads();

        int my_index = atomicAdd(&index, 1) % 1000;
        return shared_randoms[my_index];
    #else
      struct SharedRandomQueue {
          std::array<uint64_t, 10000> data;
          std::atomic<bool> initialized{false};

          SharedRandomQueue() {
              initialize();
          }

          void initialize() {
              std::cout << "Initializing shared random queue" << std::endl;
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
    #endif
    }


    #define DEFINE_COMPARISON(op) \
        DACE_HOST_DEVICE bool operator op (const float32sr& other) const { \
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
        DACE_HOST_DEVICE bool operator op (const float& other) const { \
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
    DACE_HOST_DEVICE float32sr operator=(const float& v) {
        value = v;
        return *this;
    }

    DACE_HOST_DEVICE float32sr operator=(const double& v) {
        value = stochastic_round(v);
        return *this;
    }

    DACE_HOST_DEVICE float32sr operator=(const int& v) {
        value = stochastic_round(static_cast<double>(v));
        return *this;
    }

    DACE_HOST_DEVICE float32sr operator=(const int64_t& v) {
        value = stochastic_round(static_cast<double>(v));
        return *this;
    }


    //==================
    // Primary operators
    //==================
    DACE_HOST_DEVICE float32sr operator+(const float32sr& other) const {
        return float32sr(stochastic_round(static_cast<double>(value) + static_cast<double>(other.value)));
    }

    DACE_HOST_DEVICE float32sr operator*(const float32sr& other) const {
        return float32sr(stochastic_round(static_cast<double>(value) * static_cast<double>(other.value)));
    }

    DACE_HOST_DEVICE float32sr operator-(const float32sr& other) const {
        return float32sr(stochastic_round(static_cast<double>(value) - static_cast<double>(other.value)));
    }

    DACE_HOST_DEVICE float32sr operator/(const float32sr& other) const {
        return float32sr(stochastic_round(static_cast<double>(value) / static_cast<double>(other.value)));
    }

    //==================
    // Mixing stochatic types with primative types
    //==================
    // Mixing RTN + SR should yield SR
    DACE_HOST_DEVICE float operator+(const float& other) const {
        return value + other;
    }

    DACE_HOST_DEVICE float operator*(const float& other) const {
        return value * other;
    }

    DACE_HOST_DEVICE float operator-(const float& other) const {
        return value - other;
    }

    DACE_HOST_DEVICE float operator/(const float& other) const {
        return value / other;
    }

    DACE_HOST_DEVICE float operator+(const int& other) const {
        return value + other;
    }

    DACE_HOST_DEVICE float operator*(const int& other) const {
        return value * other;
    }

    DACE_HOST_DEVICE float operator-(const int& other) const {
        return value - other;
    }

    DACE_HOST_DEVICE float operator/(const int& other) const {
        return value / other;
    }





    //==================
    // Compound assignment operators
    //==================


    DACE_HOST_DEVICE float32sr& operator+=(const float32sr& other) {
        value = stochastic_round(static_cast<double>(value) + static_cast<double>(other.value));
        return *this;
    }

    DACE_HOST_DEVICE float32sr& operator-=(const float32sr& other) {
        value = stochastic_round(static_cast<double>(value) - static_cast<double>(other.value));
        return *this;
    }

    DACE_HOST_DEVICE float32sr& operator*=(const float32sr& other) {
        value = stochastic_round(static_cast<double>(value) * static_cast<double>(other.value));
        return *this;
    }

    DACE_HOST_DEVICE float32sr& operator/=(const float32sr& other) {
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

#ifdef __CUDACC__
#include <cub/util_type.cuh>
#include <limits>

// This functionality is required by the ICON Velocity Tendecies procedure
namespace cub {

template <>
struct Traits<dace::float32sr> {
    using PRIMITIVE = float;

    __host__ __device__ static dace::float32sr Lowest() {
        return dace::float32sr(-std::numeric_limits<float>::infinity());
    }

    __host__ __device__ static dace::float32sr Max() {
        return dace::float32sr(std::numeric_limits<float>::infinity());
    }

    __host__ __device__ static dace::float32sr Zero() {
        return dace::float32sr(0.0f);
    }

    __host__ __device__ static dace::float32sr One() {
        return dace::float32sr(1.0f);
    }

    static constexpr bool IsFloatingPoint = true;
};

}
#endif


#endif  // __DACE_SROUND_H
