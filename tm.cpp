/**
 * @file   tm.cpp
 * @author Tommy Chu <tommychu2256@gmail.com>
 * @date   22 Sep 2025
 *
 * @section DESCRIPTION
 *
 * A software transactional memory implementing TL2, based on [1] and [2].
 *
 * [1] Dice, D., Shalev, O., Shavit, N. (2006). Transactional Locking II.
 *     In: Dolev, S. (eds) Distributed Computing. DISC 2006. Lecture Notes in
 *         Computer Science, vol 4167. Springer, Berlin, Heidelberg.
 *         https://doi.org/10.1007/11864219_14
 *
 * [2] Guerraoui, R., Kapatka, M. (2010). Principles of Transactional Memory.
 *     In: Synthesis Lectures on Distributed Computing Theory. Springer Cham.
 *         https://doi.org/10.1007/978-3-031-02002-5
 *
 **/

#define _POSIX_C_SOURCE 200809L

#include "tm.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

#include <immintrin.h>

/**
 * =============================================================================
 * Macros.
 **/

/** Define a proposition as likely true.
 **/
#undef likely
#if defined(__GNUC__)
#define likely(prop) __builtin_expect((prop) ? true : false, true)
#else
#define likely(prop) (prop)
#endif

/** Define a proposition as likely false.
 **/
#undef unlikely
#if defined(__GNUC__)
#define unlikely(prop) __builtin_expect((prop) ? true : false, false)
#else
#define unlikely(prop) (prop)
#endif

/** Define a variable as unused.
 **/
#undef unused
#if defined(__GNUC__)
#define unused(variable) variable __attribute__((unused))
#else
#define unused(prop) (prop)
#endif

/** Force inline function expansion.
 **/
#undef _inline
#if defined(__GNUC__)
#define _inline __attribute__((always_inline)) inline
#else
#define _inline inline
#endif

/** CPU instruction for spin-wait loops.
 **/
static _inline void spinwait() { _mm_pause(); }

/** Determine cache line size for alignment.
#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#endif
 **/
// 64 bytes on most x86-64 CPUs
constexpr std::size_t cache_line_size = 64;

/**
 * =============================================================================
 * Versioned lock with atomic uint32_t.
 * Version is stored in the upper 31 bits, lock bit in the lowest bit.
 * Lock is acquired by setting the lowest bit to 1 using CAS.
 * Memory layout:  +-----------------------+
 *                 | 31b-version | 1b-lock |
 *                 +-----------------------+
 * Aligned to cache line size to avoid false sharing and invalidating cache
 * lines between different versioned locks.
 **/
struct alignas(cache_line_size) VersionLock {
  std::atomic_uint32_t vl;

  // deleted to prevent accidental heap allocation.
  // we rely solely on memset(0) for initialization.
  VersionLock() = delete;

  static _inline constexpr uint32_t version_of(uint32_t x) { return x >> 1; }
  static _inline constexpr bool is_locked(uint32_t x) { return x & 0b1; }

  _inline bool try_lock() {
    uint32_t snap = vl.load(std::memory_order_relaxed);
    if (unlikely(snap & 0b1))
      return false;
    return vl.compare_exchange_strong(snap, snap | 0b1,
                                      // need strong ordering for locking
                                      std::memory_order_acquire,
                                      // don't care about failure ordering
                                      std::memory_order_relaxed);
  }

  _inline void lock() {
    if (likely(try_lock()))
      return;
    while (true) { // failed to get lock immediately, spin until acquired
      while (is_locked(vl.load(std::memory_order_relaxed)))
        // someone else holds the lock, wait
        spinwait();
      // grab it now!
      if (try_lock())
        return;
    }
  }

  _inline void unlock() {
    // we own the lock, no other thread is accessing it
    // no need for atomic read-modify-write here
    uint32_t current = vl.load(std::memory_order_relaxed);
    vl.store(current & ~0b1, std::memory_order_release);
  }

  _inline void unlock_and_set(uint32_t new_version) {
    vl.store(new_version << 1, std::memory_order_release);
  }
};

using VL = VersionLock;

/**
 * =============================================================================
 * Virtual address abstraction.
 * Top 16 bits are segment index, lower 48 bits are offset within the segment.
 * First 16 bits are never 0 to avoid nullptr, so segment index is offset by 1.
 * Memory layout:  +--------------------------+
 *                 | 16b-segment | 48b-offset |
 *                 +--------------------------+
 * Boundaries:
 *   - 1st  element, 1st  segment : 0x0001000000000000
 *   - last element, 1st  segment : 0x0001FFFFFFFFFFFF
 *   - 1st  element, last segment : 0xFFFF000000000000
 *   - last element, last segment : 0xFFFFFFFFFFFFFFFF
 **/
struct VirtualAddress {
  static constexpr uint64_t OFFSET = 48;

  uint64_t va;

  constexpr VirtualAddress() = delete;
  _inline constexpr VirtualAddress(uint64_t va) : va(va) {}
  _inline constexpr VirtualAddress(uint32_t segment, uint64_t offset)
      : va(((static_cast<uint64_t>(segment) + 1) << 48) | offset) {}
  _inline constexpr uint64_t virtual_address() const { return va; }
  _inline constexpr uint32_t segment() const { return (va >> 48) - 1; }
  _inline constexpr uint64_t offset() const { return (va << 16) >> 16; }
};

using VA = VirtualAddress;

/**
 * =============================================================================
 * Segment of memory in the shared region.
 * Contains versioned locks and data area.
 * Locks and data are allocated in a single malloc call and set to zero.
 * Memory layout:  +-----------+--------------+------------+
 *                 | VL* locks : padding      | char *data |
 *                 +-----------+--------------+------------+
 * Each word of size 'align' bytes has an associated versioned lock (per-word).
 * Data is of type char* just for better pointer arithmetic.
 **/
struct Segment {
  VL *locks;
  char *data;

  Segment(size_t size, size_t align) {
    size_t word_count = size / align;
    size_t locks_size = word_count * sizeof(VL);
    size_t data_offset = (locks_size + align - 1) & ~(align - 1);
    size_t total_size = data_offset + size;
    void *raw_mem = malloc(total_size);
    if (!raw_mem)
      throw std::bad_alloc();
    std::memset(raw_mem, 0, total_size);
    locks = reinterpret_cast<VL *>(raw_mem);
    data = reinterpret_cast<char *>(raw_mem) + data_offset;
  }
  Segment(Segment const &) = delete;
  Segment &operator=(const Segment &) = delete;
  ~Segment() {
    free(locks);
    locks = nullptr;
  }
};

/**
 * =============================================================================
 * Memory region.
 * Contains segments, each with its own size.
 * First segment is preallocated at creation.
 * Memory layout:  +--------------------+------------+--
 *                 | 32bit glob_version : padding    | ...
 *                 +--------------------+------------+--
 *                       --+-------+------+----------+-------+
 *                     ... | align , size | segments , mutex |
 *                       --+-------+------+----------+-------+
 * Max 65535 segments (16 bits segment index in VA, offset by 1).
 * New segments can be added and removed dynamically.
 * Segment (de)allocation is protected by a mutex.
 * Global version is aligned to cache line size to avoid false sharing.
 **/
struct Region {
  static constexpr size_t MAX_SEGMENTS = 65535; // 2^16 - 1 segments

  // accessed frequently, align to cache line size
  alignas(cache_line_size) std::atomic_uint32_t global_version;
  alignas(cache_line_size) const size_t align;
  const size_t size;
  Segment *segments[MAX_SEGMENTS];
  std::mutex alloc_mtx;

  Region(size_t size, size_t align) : align(align), size(size) {
    std::memset(segments, 0, sizeof(segments));
    global_version.store(0, std::memory_order_relaxed);
    add_segment(size);
  }
  Region(Region const &) = delete;
  Region &operator=(Region const &) = delete;
  ~Region() {
    for (Segment *&ptr : segments) {
      if (ptr) {
        delete ptr;
        ptr = nullptr;
      }
    }
  }

  // create a new segment of given size, return the segment id.
  _inline uint32_t add_segment(size_t size) {
    std::lock_guard<std::mutex> lock(alloc_mtx);
    // find a free slot
    for (uint32_t current = 0; current < MAX_SEGMENTS; ++current) {
      if (segments[current] == nullptr) {
        segments[current] = new Segment(size, align);
        return current;
      }
    }
    // no free slot
    throw std::bad_alloc();
  }

  // delete a segment.
  _inline void remove_segment(uint32_t si) {
    std::lock_guard<std::mutex> lock(alloc_mtx);
    delete segments[si];
    segments[si] = nullptr;
  }
};

/**
 * =============================================================================
 * Interface, documented in tm.hpp.
 * tm_create: build the region and its first segment (segment 0).
 **/
shared_t tm_create(size_t size, size_t align) noexcept {
  try {
    return new Region(size, align);
  } catch (...) {
    return invalid_shared;
  }
}

/**
 * =============================================================================
 * tm_destroy: delete the region and all its segments.
 **/
void tm_destroy(shared_t shared) noexcept {
  delete static_cast<Region *>(shared);
}

/**
 * =============================================================================
 * tm_start: the first segment is always segment 0.
 **/
void *tm_start(shared_t unused(shared)) noexcept {
  return reinterpret_cast<void *>(VA(0, 0).virtual_address());
}

/**
 * =============================================================================
 * tm_size: size of the first segment.
 **/
size_t tm_size(shared_t shared) noexcept {
  return static_cast<Region *>(shared)->size;
}

/**
 * =============================================================================
 * tm_align: size of a word.
 **/
size_t tm_align(shared_t shared) noexcept {
  return static_cast<Region *>(shared)->align;
}
