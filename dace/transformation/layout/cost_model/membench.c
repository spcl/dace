/* Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved. */
/*
 * Self-timed memory microbenchmarks that parametrize the LogP/LogGP memory model: a pointer chase
 * for the latency L, and a STREAM triad for the per-byte gap G.
 *
 * This is C and not numpy for one reason that is not about constant factors: a pointer chase
 * requires MLP=1 -- the next address must BE the value the previous load returned. numpy cannot
 * express that. A Python loop costs ~50-100ns per step, the same order as the DRAM latency being
 * measured; vectorizing it turns the chase into a gather at MLP=width, which measures throughput,
 * not latency. The invariant IS the benchmark.
 *
 * The kernels self-time a window and hand back nanoseconds, so the ~1us ctypes call overhead sits
 * entirely outside the measured region.
 */
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>

#define CHASE_MAX_CHAINS 32u
#define BANK_MIX_MASK 7u   /* 8 L1 data-cache banks */
#define BANK_MIX_SHIFT 3u  /* banks are indexed by address bits 5:3 */

typedef struct {
    unsigned char *arena;
    size_t arena_bytes;
    size_t stride;
    size_t nr_elts;
    unsigned nchain;
    void *starts[CHASE_MAX_CHAINS];
} chase_handle;

static uint64_t now_ns(void)
{
    /* CLOCK_MONOTONIC, not rdtsc: the invariant TSC ticks at the NOMINAL frequency regardless of
     * turbo, so a TSC delta is true time but NOT core cycles -- dividing it by a turbo frequency to
     * get cycles is the classic error. clock_gettime is a vDSO read of the same counter and needs no
     * MSR, so there is nothing to gain from rdtsc here. */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t major_faults(void)
{
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return (uint64_t)usage.ru_majflt;
}

/* xorshift64*: deterministic across runs given a seed, and far cheaper than rand(). */
static uint64_t rng_next(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ull;
}

static size_t rng_below(uint64_t *state, size_t bound)
{
    return (size_t)(rng_next(state) % (uint64_t)bound);
}

/* Address of chase element i.
 *
 * The bank mixer matters: with a plain i*stride layout every element is stride-aligned, so address
 * bits 5:3 are identical for every access and all of them hit ONE L1 data-cache bank, which shows up
 * as an inflated L1 latency. Offsetting by (i & 7) << 3 spreads the eight banks while keeping each
 * element on its own cache line (stride >= 64 guarantees that; the offset is < 64). */
static void **elem_addr(const chase_handle *h, size_t i)
{
    return (void **)(h->arena + i * h->stride + ((i & BANK_MIX_MASK) << BANK_MIX_SHIFT));
}

/*
 * Build `nchain` disjoint pointer chains, each a single Hamiltonian cycle over its own elements.
 *
 * Method is multichase's: a Fisher-Yates permutation plus its INVERSE, linking each element to its
 * successor in permutation order. That construction yields exactly ONE cycle covering every element.
 * The tempting alternative -- fill an array with self-pointers and shuffle it -- is what X-Mem does,
 * and X-Mem's own paper concedes it "may occasionally form a small cycle that traps the kernel
 * function, effectively shrinking the intended working set size". A trapped chase silently reports a
 * cache latency labelled DRAM, so it is not used here; chase_verify() proves the cycle instead.
 *
 * Returns NULL on failure.
 */
chase_handle *chase_setup(size_t arena_bytes, size_t stride, unsigned nchain, unsigned seed)
{
    if (stride < 64 || nchain == 0 || nchain > CHASE_MAX_CHAINS)
        return NULL;

    chase_handle *h = (chase_handle *)calloc(1, sizeof(chase_handle));
    if (!h)
        return NULL;

    h->arena = (unsigned char *)mmap(NULL, arena_bytes, PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (h->arena == MAP_FAILED) {
        free(h);
        return NULL;
    }
    /* 2 MiB pages: a 512 MiB arena is then 256 pages, which fits entirely in the L2 DTLB, so a fully
     * random chase over the whole arena takes ZERO TLB misses. That is what lets the chase be fully
     * random (the strongest prefetcher defeat) without a page-walk term contaminating L. The caller
     * must verify AnonHugePages actually materialized -- madvise is advisory. */
    (void)madvise(h->arena, arena_bytes, MADV_HUGEPAGE);
    memset(h->arena, 0, arena_bytes); /* fault in now, so the timed window never page-faults */

    h->arena_bytes = arena_bytes;
    h->stride = stride;
    h->nchain = nchain;
    h->nr_elts = arena_bytes / stride;
    if (h->nr_elts < nchain * 2) {
        munmap(h->arena, arena_bytes);
        free(h);
        return NULL;
    }

    size_t per_chain = h->nr_elts / nchain;
    size_t *perm = (size_t *)malloc(per_chain * sizeof(size_t));
    size_t *perm_inverse = (size_t *)malloc(per_chain * sizeof(size_t));
    if (!perm || !perm_inverse) {
        free(perm);
        free(perm_inverse);
        munmap(h->arena, arena_bytes);
        free(h);
        return NULL;
    }

    uint64_t rng = seed ? (uint64_t)seed : 1ull;
    for (unsigned c = 0; c < nchain; ++c) {
        size_t base = (size_t)c * per_chain;

        /* Fisher-Yates, inside-out: perm is a permutation of [base, base+per_chain). */
        for (size_t i = 0; i < per_chain; ++i) {
            size_t t = rng_below(&rng, i + 1);
            perm[i] = perm[t];
            perm[t] = base + i;
        }
        for (size_t i = 0; i < per_chain; ++i)
            perm_inverse[perm[i] - base] = i;

        /* Link element i to the successor of i in permutation order. Walking the permutation in its
         * own index order this way closes exactly one cycle over all per_chain elements. */
        for (size_t i = 0; i < per_chain; ++i) {
            size_t next = perm_inverse[i] + 1;
            if (next == per_chain)
                next = 0;
            *elem_addr(h, base + i) = (void *)elem_addr(h, perm[next]);
        }
        h->starts[c] = (void *)elem_addr(h, base);
    }

    free(perm);
    free(perm_inverse);
    return h;
}

/*
 * Prove every chain is ONE cycle over exactly its own elements: walk from the start and require the
 * walk to return to the start after exactly per_chain hops, never earlier. This is the decisive
 * correctness gate for the whole benchmark -- a chase trapped in a sub-cycle looks perfectly healthy
 * and reports a cache latency as if it were DRAM.
 *
 * Returns 0 on success, non-zero on a malformed chain.
 */
int chase_verify(const chase_handle *h)
{
    if (!h)
        return -1;
    size_t per_chain = h->nr_elts / h->nchain;
    for (unsigned c = 0; c < h->nchain; ++c) {
        void *p = h->starts[c];
        for (size_t hops = 1; hops <= per_chain; ++hops) {
            p = *(void **)p;
            if (p == h->starts[c])
                return (hops == per_chain) ? 0 : -2; /* closed early: a sub-cycle */
        }
        return -3; /* never closed */
    }
    return 0;
}

/* Evict the arena: construction leaves it hot, and an unflushed first window measures cache. */
void chase_flush(chase_handle *h)
{
    if (!h)
        return;
    size_t bytes = 64u << 20;
    volatile unsigned char *scratch = (volatile unsigned char *)malloc(bytes);
    if (!scratch)
        return;
    for (size_t i = 0; i < bytes; i += 64)
        scratch[i] = (unsigned char)i;
    free((void *)scratch);
}

/*
 * Chase `hops` times per chain and report the elapsed nanoseconds and any major faults taken.
 *
 * Anti-elision uses an inline-asm consume of the final pointers rather than declaring them volatile:
 * a volatile pointer forces a stack reload every step, putting store-forwarding on the critical path
 * and inflating the L1 measurement by several cycles.
 *
 * Returns the number of hops actually performed (hops * nchain).
 */
uint64_t chase_run_timed(chase_handle *h, uint64_t hops, uint64_t *time_ns, uint64_t *majflt)
{
    if (!h)
        return 0;
    void *p[CHASE_MAX_CHAINS];
    for (unsigned c = 0; c < h->nchain; ++c)
        p[c] = h->starts[c];

    unsigned nchain = h->nchain;
    uint64_t faults_before = major_faults();
    uint64_t start = now_ns();
    if (nchain == 1) {
        /* nchain==1 is where L itself is defined, so it gets a dedicated body: keeping the pointer in
         * a LOCAL (not an array slot) compiles to a bare `mov (%rax),%rax` loop. Going through p[]
         * makes gcc reload and spill the slot every hop, putting a store-forward round trip on the
         * critical path -- ~2% of a 95ns DRAM hop, but multiples of an L1 hop. */
        void *q = p[0];
        for (uint64_t i = 0; i < hops; ++i)
            q = *(void **)q; /* the whole benchmark: one dependent load */
        p[0] = q;
    } else {
        /* nchain>1 measures the latency-vs-MLP curve, where the per-hop spill is small against the
         * exposed latency and the chains' independence is what is being measured. */
        for (uint64_t i = 0; i < hops; ++i)
            for (unsigned c = 0; c < nchain; ++c)
                p[c] = *(void **)p[c];
    }
    uint64_t end = now_ns();
    uint64_t faults_after = major_faults();

    for (unsigned c = 0; c < nchain; ++c)
        __asm__ volatile("" : : "r"(p[c]) : "memory");

    if (time_ns)
        *time_ns = end - start;
    if (majflt)
        *majflt = faults_after - faults_before;
    return hops * (uint64_t)nchain;
}

void chase_teardown(chase_handle *h)
{
    if (!h)
        return;
    munmap(h->arena, h->arena_bytes);
    free(h);
}

/*
 * STREAM triad: a[i] = b[i] + q * c[i].
 *
 * The caller converts iterations to HARDWARE bytes. STREAM counts 24 B/iter (read b, read c, write
 * a) but the store to a cold line first pulls that line in -- a read-for-ownership -- so the real
 * DRAM traffic is 32 B/iter, a factor of 4/3. Fitting G to STREAM's printed number makes G 33% too
 * optimistic.
 */
void triad_run_timed(double *a, const double *b, const double *c, double q, size_t n, uint64_t *time_ns)
{
    uint64_t start = now_ns();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        a[i] = b[i] + q * c[i];
    uint64_t end = now_ns();
    if (time_ns)
        *time_ns = end - start;
    __asm__ volatile("" : : "r"(a) : "memory");
}
