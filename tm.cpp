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
 * Bloom filter for fast membership test with false positives.
 * Description: https://en.wikipedia.org/wiki/Bloom_filter
 *
 * Inspired by the Murmur3 implementation from:
 * : https://gist.github.com/kevinmoran/471480b1e20a19b0687d81b75fd801c8
 * : https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 **/
constexpr size_t BLOOM_BITS = 4096; // 2^12 bits, 512 bytes
constexpr size_t BLOOM_MASK = BLOOM_BITS - 1;
constexpr size_t BLOOM_WORD_COUNT = BLOOM_BITS / 64;
constexpr size_t CACHE_LINE_SHIFT = 2;
struct alignas(cache_line_size) BloomFilter {
  uint64_t data[BLOOM_WORD_COUNT];

  BloomFilter() { clear(); }

  static _inline uint32_t fmix(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    // h ^= h >> 13;
    // h *= 0xc2b2ae35;
    return h ^= h >> 16;
  }

  static _inline void key_bits(const void *ptr, uint32_t &k1, uint32_t &k2) {
    uint32_t val = reinterpret_cast<uintptr_t>(ptr);
    val >>= CACHE_LINE_SHIFT; // ignore lower bits within cache line
    uint32_t h = fmix(val);
    k1 = h & BLOOM_MASK; // 12 bits per key
    k2 = (h >> 12) & BLOOM_MASK;
  }

  _inline void add(const void *addr_ptr) {
    uint32_t k1, k2;
    key_bits(addr_ptr, k1, k2);
    // d[kx / 64] |= (1    << (kx % 64))
    data[k1 >> 6] |= (1ULL << (k1 & 63));
    data[k2 >> 6] |= (1ULL << (k2 & 63));
  }

  _inline bool possibly_contains(const void *addr_ptr) const {
    uint32_t k1, k2;
    key_bits(addr_ptr, k1, k2);
    uint64_t bit1 = (data[k1 >> 6] >> (k1 & 63));
    uint64_t bit2 = (data[k2 >> 6] >> (k2 & 63));
    return (bit1 & bit2) & 1;
  }

  _inline void clear() { std::memset(data, 0, sizeof(data)); }
};

using BF = BloomFilter;

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
 * (disabled to save memory, there is one lock per word)
   alignas(cache_line_size)
 **/
struct VersionLock {
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
 * Transaction context.
 * Contains read set, write set, write buffer, bloom filter, versions.
 * Memory layout:  +---------------+--------------+--------------------+--
 *                 | read only bit : read version | write bloom filter | ...
 *                 +---------------+--------------+--------------------+--
 *               --+---------------+--------------+--------------------+--
 *             ... | read set      , write set    , write buffer       | ...
 *               --+---------------+--------------+--------------------+--
 *                               --+--------------+--------------------+
 *                             ... | allocated    / freed segments     |
 *                               --+--------------+--------------------+
 * ReadLog: contains the versioned lock of the read word.
 * WriteLog: contains the versioned lock, target address, and offset in wbuffer.
 * Write buffer: contiguous buffer for all write data.
 * Allocated / freed segments: list of segment ids allocated / freed in tx.
 * Bloom filter: for fast membership test in write set.
 **/
struct Transaction {
  using ReadLog = VL *;
  struct WriteLog {
    VL *vl;
    void *addr;
    uint64_t offset;
  };

  alignas(cache_line_size) bool read_only;
  uint32_t read_version;
  BF write_bloom;
  std::vector<ReadLog> read_set;
  std::vector<WriteLog> write_set;
  std::vector<char> write_buffer;
  std::vector<uint32_t> allocated_segments;
  std::vector<uint32_t> freed_segments;

  Transaction() : read_only(false), read_version(0) {
    read_set.reserve(256);
    write_set.reserve(64);
    write_buffer.reserve(1024);
    allocated_segments.reserve(8);
    freed_segments.reserve(8);
  }
  Transaction(Transaction const &) = delete;
  Transaction &operator=(Transaction const &) = delete;

  void reset(bool is_ro, uint32_t rv) {
    read_only = is_ro;
    read_version = rv;
    write_bloom.clear();
    read_set.clear();
    write_set.clear();
    write_buffer.clear();
    allocated_segments.clear();
    freed_segments.clear();
  }
};

using TX = Transaction;

/**
 * =============================================================================
 * End of a transaction.
 **/
// abort: the segments it allocated were never visible to other transactions.
static bool abort_tx(Region *region, TX *tx) {
  for (uint32_t si : tx->allocated_segments)
    region->remove_segment(si);
  tx->allocated_segments.clear();
  return false;
}
// commit: for now, the segments it freed are only deleted with the region.
static bool commit_tx(Region *unused(region), TX *unused(tx)) { return true; }

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

/**
 * =============================================================================
 * tm_begin: reuse the context of the thread and sample the read version.
 **/
tx_t tm_begin(shared_t shared, bool is_ro) noexcept {
  Region *region = static_cast<Region *>(shared);
  static thread_local TX tx; // no nested transactions
  try {
    uint32_t rv = region->global_version.load();
    tx.reset(is_ro, rv);
    return reinterpret_cast<tx_t>(&tx);
  } catch (...) {
    return invalid_tx;
  }
}

/**
 * =============================================================================
 * tm_end: commit, i.e. lock the write set, take a write version, validate the
 *         read set, write back and unlock.
 **/
bool _tm_end(shared_t shared, tx_t _tx) {
  Region *region = static_cast<Region *>(shared);
  TX *tx = reinterpret_cast<TX *>(_tx);
  if (tx->read_only || tx->write_set.empty()) {
    // yay, shortcut to glory
    return commit_tx(region, tx);
  }
  std::sort(/* sort locks by address to prevent deadlock */
            /* then by time of write, the last write to a word must win */
            tx->write_set.begin(), tx->write_set.end(),
            [](const TX::WriteLog &a, const TX::WriteLog &b) {
              return a.vl != b.vl ? a.vl < b.vl : a.offset < b.offset;
            });
  // acquire all write locks
  VL *prev_lock = nullptr;
  for (const TX::WriteLog &wl : tx->write_set) {
    if (wl.vl == prev_lock)
      continue; // skip duplicate locks
    wl.vl->lock();
    prev_lock = wl.vl;
  }
  // increment global version
  uint32_t write_version = region->global_version.fetch_add(1) + 1;
  // no writes can happen between reads and writes (we hold locks)
  if (tx->read_version + 1 != write_version) {
    // concurrent writes happened
    for (const VL *rl : tx->read_set) {
      uint32_t snap = rl->vl.load(std::memory_order_acquire);
      bool locked_by_me = false;
      if (VL::is_locked(snap)) {
        // is it in my write set?
        auto it = std::lower_bound(
            tx->write_set.begin(), tx->write_set.end(), rl,
            [](const TX::WriteLog &wl, const VL *val) { return wl.vl < val; });
        if (it != tx->write_set.end() && it->vl == rl)
          locked_by_me = true;
      }
      if (unlikely((!locked_by_me && VL::is_locked(snap)) ||
                   VL::version_of(snap) > tx->read_version)) {
        // vl being modified, or is outdated
        // release all locks in write set and abort
        prev_lock = nullptr;
        for (const TX::WriteLog &wl : tx->write_set) {
          if (wl.vl == prev_lock)
            continue;
          wl.vl->unlock();
          prev_lock = wl.vl;
        }
        return abort_tx(region, tx);
      }
    }
  }
  // commit writes
  for (const TX::WriteLog &wl : tx->write_set)
    std::memcpy(wl.addr, tx->write_buffer.data() + wl.offset, region->align);
  prev_lock = nullptr;
  for (const TX::WriteLog &wl : tx->write_set) {
    if (wl.vl == prev_lock)
      continue;
    wl.vl->unlock_and_set(write_version);
    prev_lock = wl.vl;
  }
  return commit_tx(region, tx);
}
bool tm_end(shared_t shared, tx_t _tx) noexcept {
  try {
    return _tm_end(shared, _tx);
  } catch (...) {
    return false;
  }
}

/**
 * =============================================================================
 * Read of a single word (helpers of tm_read).
 **/
// copy a word out of shared memory: fails if it is locked, newer than the read
// version, or modified during the copy (lock sampled before and after).
inline bool read_word(Region &region, TX &tx, VL *source_lock, const void *from,
                      void *to) {
  uint32_t snap = source_lock->vl.load(std::memory_order_acquire);
  if (unlikely(VL::is_locked(snap) || VL::version_of(snap) > tx.read_version))
    return false;
  std::memcpy(to, from, region.align);
  std::atomic_thread_fence(std::memory_order_acquire);
  return likely(snap == source_lock->vl.load(std::memory_order_relaxed));
}
inline bool read_word_rw(Region &region, TX &tx, VL *source_lock,
                         const void *from, void *to) {
  if (!tx.write_set.empty() && tx.write_bloom.possibly_contains(from)) {
    for (auto it = tx.write_set.rbegin(); it != tx.write_set.rend(); ++it) {
      if (it->addr == from) {
        std::memcpy(to, tx.write_buffer.data() + it->offset, region.align);
        return true;
      }
    }
  }
  // read from shared memory
  if (!read_word(region, tx, source_lock, from, to))
    return false;
  tx.read_set.push_back(source_lock);
  return true;
}
/**
 * =============================================================================
 * tm_read: word by word, from the write set if the transaction wrote the word,
 *          from shared memory otherwise.
 **/
inline bool _tm_read(shared_t shared, tx_t _tx, void const *source, size_t size,
                     void *target) {
  Region *region = static_cast<Region *>(shared);
  TX *tx = reinterpret_cast<TX *>(_tx);
  VA source_va(reinterpret_cast<uint64_t>(source));
  Segment *segment = region->segments[source_va.segment()];
  // compute counts
  size_t align = region->align;
  size_t word_count = size / align;
  size_t word_index = source_va.offset() / align;
  // running pointers
  const char *__restrict__ _from = segment->data + source_va.offset();
  char *__restrict__ _to = static_cast<char *>(target);
  VL *__restrict__ _lock = segment->locks + word_index;
  if (tx->read_only) {
    for (size_t wi = 0; wi < word_count; ++wi) {
      if (!read_word(*region, *tx, _lock, _from, _to))
        return false;
      // move pointers
      _from += align;
      _to += align;
      _lock++;
    }
    return true;
  }
  for (size_t wi = 0; wi < word_count; ++wi) {
    if (!read_word_rw(*region, *tx, _lock, _from, _to))
      return false;
    // move pointers
    _from += align;
    _to += align;
    _lock++;
  }
  return true;
}
bool tm_read(shared_t shared, tx_t _tx, void const *source, size_t size,
             void *target) noexcept {
  try {
    if (likely(_tm_read(shared, _tx, source, size, target)))
      return true;
  } catch (...) {
  }
  return abort_tx(static_cast<Region *>(shared), reinterpret_cast<TX *>(_tx));
}

/**
 * =============================================================================
 * tm_write: buffer the words in the write set, shared memory is left untouched
 *           until commit.
 **/
inline bool _tm_write(shared_t shared, tx_t _tx, void const *source,
                      size_t size, void *target) {
  Region *region = static_cast<Region *>(shared);
  TX *tx = reinterpret_cast<TX *>(_tx);
  VA target_va(reinterpret_cast<uint64_t>(target));
  Segment *segment = region->segments[target_va.segment()];
  // compute counts
  size_t align = region->align;
  size_t word_count = size / align;
  size_t word_index = target_va.offset() / align;
  // buffered write
  size_t buffer_offset = tx->write_buffer.size();
  tx->write_buffer.resize(buffer_offset + size);
  char *buffer_ptr = tx->write_buffer.data() + buffer_offset;
  // running pointers
  const char *__restrict__ _from = static_cast<const char *>(source);
  char *__restrict__ _to = segment->data + target_va.offset();
  VL *__restrict__ _lock = segment->locks + word_index;
  for (size_t wi = 0; wi < word_count; ++wi) {
    tx->write_bloom.add(_to);
    std::memcpy(buffer_ptr, _from, align);
    tx->write_set.push_back({_lock, _to, buffer_offset});
    // move pointers
    buffer_offset += align;
    buffer_ptr += align;
    _from += align;
    _to += align;
    _lock++;
  }
  return true;
}
bool tm_write(shared_t shared, tx_t _tx, void const *source, size_t size,
              void *target) noexcept {
  try {
    return _tm_write(shared, _tx, source, size, target);
  } catch (...) {
    return abort_tx(static_cast<Region *>(shared), reinterpret_cast<TX *>(_tx));
  }
}

/**
 * =============================================================================
 * tm_alloc: add the segment right away, it is removed if the transaction
 *           aborts.
 **/
Alloc tm_alloc(shared_t shared, tx_t _tx, size_t size, void **target) noexcept {
  Region *region = static_cast<Region *>(shared);
  TX *tx = reinterpret_cast<TX *>(_tx);
  try {
    uint32_t si = region->add_segment(size);
    tx->allocated_segments.push_back(si);
    *target = reinterpret_cast<void *>(VA(si, 0).virtual_address());
    return Alloc::success;
  } catch (...) {
    return Alloc::nomem;
  }
}

/**
 * =============================================================================
 * tm_free: only record the segment, for now it is deleted with the region.
 **/
bool tm_free(shared_t shared, tx_t _tx, void *target) noexcept {
  TX *tx = reinterpret_cast<TX *>(_tx);
  VA target_va(reinterpret_cast<uint64_t>(target));
  try {
    tx->freed_segments.push_back(target_va.segment());
  } catch (...) {
    return abort_tx(static_cast<Region *>(shared), tx);
  }
  return true;
}
