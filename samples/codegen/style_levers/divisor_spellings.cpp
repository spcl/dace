#include <cstdint>
// SEMANTICALLY IDENTICAL: every function computes b % 4. Only the spelling of the constant differs.
uint64_t v0_literal(uint64_t b)      { return b % 4; }
uint64_t v1_const(uint64_t b)        { const uint64_t a = 4; return b % a; }
uint64_t v2_constexpr(uint64_t b)    { constexpr uint64_t a = 4; return b % a; }
// --- obfuscations that try to hide that a==4 ---
uint64_t v3_volatile(uint64_t b)     { volatile uint64_t a = 4; return b % a; }              // volatile load: opaque
uint64_t v4_byref(uint64_t b, const uint64_t& a) { return b % a; }                            // const& param
static uint64_t launder(uint64_t x){ asm("" : "+r"(x)); return x; }                           // optimization barrier
uint64_t v5_asm(uint64_t b)          { uint64_t a = launder(4); return b % a; }               // asm-laundered 4
uint64_t v6_runtime(uint64_t b, uint64_t a) { return b % a; }                                  // honest runtime
