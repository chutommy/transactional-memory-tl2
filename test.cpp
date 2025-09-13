/**
 * @file   test.cpp
 * @author Tommy Chu <tommychu2256@gmail.com>
 *
 * @section DESCRIPTION
 *
 * Test suite of the transactional memory.
 **/

#include "tm.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
  return 0;
}
