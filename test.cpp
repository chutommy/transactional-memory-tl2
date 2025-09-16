/**
 * @file   test.cpp
 * @author Tommy Chu <tommychu2256@gmail.com>
 *
 * @section DESCRIPTION
 *
 * Test suite of the transactional memory.
 * Single-threaded tests check the interface (read, write, alloc, free), then
 * concurrent tests check that transactions are atomic and isolated.
 **/

#include "tm.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

/**
 * =============================================================================
 * Helpers.
 **/

/** Stop the whole run on a failed expectation (callable from any thread).
 **/
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,    \
                   #cond);                                                     \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

/** Most tests use regions of 8-byte words, holding integers or addresses.
 **/
using word_t = intptr_t;
constexpr size_t WORD = sizeof(word_t);

/** Shared address 'bytes' after 'base' (shared addresses are opaque, so no
 *  pointer arithmetic on them).
 **/
static void *at(void *base, size_t bytes) {
  return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(base) + bytes);
}

/** Running transaction, throws Abort as soon as an operation aborts it.
 **/
struct Abort {};
struct Tx {
  shared_t tm;
  tx_t id;

  void read(const void *source, size_t size, void *target) {
    if (!tm_read(tm, id, source, size, target))
      throw Abort{};
  }
  void write(const void *source, size_t size, void *target) {
    if (!tm_write(tm, id, source, size, target))
      throw Abort{};
  }
  void *alloc(size_t size) {
    void *segment = nullptr;
    Alloc result = tm_alloc(tm, id, size, &segment);
    if (result == Alloc::abort)
      throw Abort{};
    CHECK(result == Alloc::success);
    return segment;
  }
  void free(void *segment) {
    if (!tm_free(tm, id, segment))
      throw Abort{};
  }

  // single word shortcuts
  word_t get(void *address) {
    word_t value;
    read(address, WORD, &value);
    return value;
  }
  void set(void *address, word_t value) { write(&value, WORD, address); }
};

/** Run 'body' in a transaction, again and again until it commits.
 **/
template <class Body>
static void atomically(shared_t tm, bool is_ro, Body &&body) {
  while (true) {
    Tx tx{tm, tm_begin(tm, is_ro)};
    CHECK(tx.id != invalid_tx);
    try {
      body(tx);
    } catch (Abort const &) {
      continue;
    }
    if (tm_end(tm, tx.id))
      return;
  }
}

/** Run 'work(thread index)' in 'count' threads released at the same time.
 **/
template <class Work> static void parallel(size_t count, Work &&work) {
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  for (size_t index = 0; index < count; ++index)
    threads.emplace_back([&, index] {
      while (!go.load())
        std::this_thread::yield();
      work(index);
    });
  go.store(true);
  for (std::thread &thread : threads)
    thread.join();
}

static size_t thread_count() {
  return std::max(2u, std::thread::hardware_concurrency());
}

/**
 * =============================================================================
 * Single-threaded: a region starts zeroed, a transaction reads its own writes,
 * and committed writes are visible to the next transactions, whatever the
 * alignment and the number of words accessed at once.
 **/
static void test_read_write() {
  constexpr size_t SIZE = 256;
  std::minstd_rand rng(1);
  for (size_t align : {8, 16, 32, 64, 128, 256}) {
    shared_t tm = tm_create(SIZE, align);
    CHECK(tm != invalid_shared);
    void *start = tm_start(tm);
    CHECK(tm_size(tm) == SIZE);
    CHECK(tm_align(tm) == align);
    CHECK(reinterpret_cast<uintptr_t>(start) % align == 0);

    uint8_t expected[SIZE] = {}, own[SIZE], rw[SIZE], ro[SIZE];
    atomically(tm, true, [&](Tx &tx) { tx.read(start, SIZE, ro); });
    CHECK(std::memcmp(expected, ro, SIZE) == 0);

    for (size_t size = align; size <= SIZE; size *= 2) {
      for (uint8_t &byte : expected)
        byte = static_cast<uint8_t>(rng());
      atomically(tm, false, [&](Tx &tx) {
        tx.write(expected, size, start);
        tx.read(start, size, own);
      });
      atomically(tm, false, [&](Tx &tx) { tx.read(start, size, rw); });
      atomically(tm, true, [&](Tx &tx) { tx.read(start, size, ro); });
      CHECK(std::memcmp(expected, own, size) == 0);
      CHECK(std::memcmp(expected, rw, size) == 0);
      CHECK(std::memcmp(expected, ro, size) == 0);
    }
    tm_destroy(tm);
  }
}

/**
 * =============================================================================
 * Single-threaded: regions are independent from each other.
 **/
static void test_multiple_regions() {
  constexpr size_t REGIONS = 128, WORDS = 32, ROUNDS = 16;
  std::minstd_rand rng(1);
  std::vector<shared_t> tms(REGIONS);
  for (shared_t &tm : tms)
    CHECK((tm = tm_create(WORDS * WORD, WORD)) != invalid_shared);

  for (size_t round = 0; round < ROUNDS; ++round) {
    size_t offset = (rng() % WORDS) * WORD;
    for (size_t i = 0; i < REGIONS; ++i)
      atomically(tms[i], false, [&](Tx &tx) {
        tx.set(at(tm_start(tms[i]), offset), i * ROUNDS + round);
      });
    for (size_t i = 0; i < REGIONS; ++i) {
      word_t rw, ro;
      void *address = at(tm_start(tms[i]), offset);
      atomically(tms[i], false, [&](Tx &tx) { rw = tx.get(address); });
      atomically(tms[i], true, [&](Tx &tx) { ro = tx.get(address); });
      CHECK(rw == static_cast<word_t>(i * ROUNDS + round));
      CHECK(ro == rw);
    }
  }
  for (shared_t tm : tms)
    tm_destroy(tm);
}

/**
 * =============================================================================
 * Single-threaded: many segments can be allocated, start zeroed, are
 * independent from each other and can be freed.
 **/
static void test_many_segments() {
  constexpr size_t SEGMENTS = 1024;
  shared_t tm = tm_create(WORD, WORD);
  CHECK(tm != invalid_shared);
  std::vector<void *> segments(SEGMENTS);

  for (size_t i = 0; i < SEGMENTS; ++i)
    atomically(tm, false, [&](Tx &tx) {
      segments[i] = tx.alloc(2 * WORD);
      tx.set(segments[i], i);
    });
  for (size_t i = 0; i < SEGMENTS; ++i) {
    word_t written, untouched;
    atomically(tm, true, [&](Tx &tx) {
      written = tx.get(segments[i]);
      untouched = tx.get(at(segments[i], WORD));
    });
    CHECK(written == static_cast<word_t>(i));
    CHECK(untouched == 0);
  }
  for (size_t i = 0; i < SEGMENTS; ++i)
    atomically(tm, false, [&](Tx &tx) { tx.free(segments[i]); });
  tm_destroy(tm);
}

/**
 * =============================================================================
 * Concurrent: every thread repeatedly reads a shared counter in a read-only
 * transaction, then decrements it in a read-write transaction.
 * The counter must never go up between the two (isolation), and must end at
 * exactly 0 (no lost update).
 **/
static void test_counter() {
  constexpr size_t ROUNDS = 10000;
  const size_t threads = thread_count();
  shared_t tm = tm_create(WORD, WORD);
  CHECK(tm != invalid_shared);
  void *counter = tm_start(tm);
  atomically(tm, false, [&](Tx &tx) { tx.set(counter, threads * ROUNDS); });

  parallel(threads, [&](size_t) {
    for (size_t round = 0; round < ROUNDS; ++round) {
      word_t before, value;
      atomically(tm, true, [&](Tx &tx) { before = tx.get(counter); });
      atomically(tm, false, [&](Tx &tx) {
        value = tx.get(counter);
        tx.set(counter, value - 1);
      });
      CHECK(value <= before);
    }
  });

  word_t left;
  atomically(tm, true, [&](Tx &tx) { left = tx.get(counter); });
  CHECK(left == 0);
  tm_destroy(tm);
}

/**
 * =============================================================================
 * Entry point: run every test, stop at the first failure.
 **/
static void run(const char *name, void (*test)()) {
  std::printf("%-20s ", name);
  std::fflush(stdout);
  test();
  std::printf("ok\n");
}

int main() {
  run("read/write", test_read_write);
  run("multiple regions", test_multiple_regions);
  run("many segments", test_many_segments);
  run("counter", test_counter);
  return 0;
}
