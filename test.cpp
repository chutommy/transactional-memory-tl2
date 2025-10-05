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
 * Single-threaded: when a transaction writes a word several times, it reads
 * back and commits the last value written.
 **/
static void test_overwrite() {
  constexpr size_t WORDS = 256, PASSES = 3;
  shared_t tm = tm_create(WORDS * WORD, WORD);
  CHECK(tm != invalid_shared);
  void *start = tm_start(tm);

  size_t stale = 0;
  atomically(tm, false, [&](Tx &tx) {
    stale = 0;
    for (size_t pass = 0; pass < PASSES; ++pass)
      for (size_t i = 0; i < WORDS; ++i) {
        tx.set(at(start, i * WORD), pass * WORDS + i);
        stale += tx.get(at(start, i * WORD)) != word_t(pass * WORDS + i);
      }
  });
  CHECK(stale == 0);
  atomically(tm, true, [&](Tx &tx) {
    stale = 0;
    for (size_t i = 0; i < WORDS; ++i)
      stale += tx.get(at(start, i * WORD)) != word_t((PASSES - 1) * WORDS + i);
  });
  CHECK(stale == 0);
  tm_destroy(tm);
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
 * Single-threaded: large segments, either the first of a region or allocated
 * in a transaction, can be written all over and their memory is given back
 * (each round would otherwise keep 2 x LARGE bytes).
 **/
static void test_large_segments() {
  constexpr size_t LARGE = 128 << 20, PAGE = 4096, ROUNDS = 4;
  std::minstd_rand rng(1);
  for (size_t round = 0; round < ROUNDS; ++round) {
    shared_t tm = tm_create(LARGE, WORD);
    CHECK(tm != invalid_shared);
    void *allocated;
    atomically(tm, false, [&](Tx &tx) { allocated = tx.alloc(LARGE); });

    for (void *segment : {tm_start(tm), allocated}) {
      atomically(tm, false, [&](Tx &tx) {
        for (size_t offset = 0; offset < LARGE; offset += PAGE)
          tx.set(at(segment, offset), round + offset);
      });
      for (size_t i = 0; i < 16; ++i) {
        size_t offset = (rng() % (LARGE / PAGE)) * PAGE;
        word_t value;
        atomically(tm, true,
                   [&](Tx &tx) { value = tx.get(at(segment, offset)); });
        CHECK(value == static_cast<word_t>(round + offset));
      }
    }
    atomically(tm, false, [&](Tx &tx) { tx.free(allocated); });
    tm_destroy(tm);
  }
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
 * Concurrent: a shared word points to a segment, whose words all hold the
 * number of times it was replaced. Threads either read the whole segment in a
 * read-only transaction, or replace it by a new one and free it.
 * A reader must never see a segment half written or already freed, no
 * replacement can be lost, and freed segments must be recycled as there are
 * more replacements than a region can hold segments.
 **/
static void test_swap() {
  constexpr size_t WORDS = 16, ROUNDS = 200000;
  const size_t threads = thread_count();
  shared_t tm = tm_create(WORD, WORD);
  CHECK(tm != invalid_shared);
  void *head = tm_start(tm);
  std::atomic<word_t> swaps{0};

  parallel(threads, [&](size_t index) {
    std::minstd_rand rng(index + 1);
    for (size_t round = 0; round < ROUNDS / threads; ++round) {
      if (rng() % 2) {
        atomically(tm, false, [&](Tx &tx) {
          void *current = reinterpret_cast<void *>(tx.get(head));
          void *segment = tx.alloc(WORDS * WORD);
          word_t count = current ? tx.get(current) + 1 : 1;
          for (size_t i = 0; i < WORDS; ++i)
            tx.set(at(segment, i * WORD), count);
          tx.set(head, reinterpret_cast<word_t>(segment));
          if (current)
            tx.free(current);
        });
        swaps.fetch_add(1);
      } else {
        bool same = true;
        atomically(tm, true, [&](Tx &tx) {
          void *current = reinterpret_cast<void *>(tx.get(head));
          word_t count = current ? tx.get(current) : 0;
          same = true;
          for (size_t i = 1; current && i < WORDS; ++i)
            same &= tx.get(at(current, i * WORD)) == count;
        });
        CHECK(same);
      }
    }
  });

  word_t count;
  atomically(tm, true, [&](Tx &tx) {
    count = tx.get(reinterpret_cast<void *>(tx.get(head)));
  });
  CHECK(count == swaps.load());
  tm_destroy(tm);
}

/**
 * =============================================================================
 * Concurrent: a bank, whose accounts live in a linked list of segments.
 * Segment layout:  +-------+------+--------+-----------+-----+-----------+
 *                  | count | next | parity | balance 0 | ... | balance N |
 *                  +-------+------+--------+-----------+-----+-----------+
 * Every thread runs a random mix of transactions:
 *   - audit    (read-only)  : sum all the balances
 *   - transfer (read-write) : move one unit between two accounts
 *   - resize   (read-write) : open or close the last account, which allocates
 *                             or frees a segment when crossing its boundary
 * Closing an account moves what it holds beyond the initial balance to the
 * 'parity' of its segment, so that every audit must observe:
 *   sum(balances) + sum(parities) == initial balance * number of accounts
 **/
static void test_bank() {
  enum : size_t { COUNT, NEXT, PARITY, BALANCES };
  const size_t threads = thread_count();
  const size_t tx_per_thread = 200000 / threads;
  const size_t capacity = 16 * threads; // accounts per segment
  const size_t target = 4 * capacity;   // accounts the bank grows to
  const word_t initial = 100;           // balance of a new account
  const size_t segment_size = (BALANCES + capacity) * WORD;

  shared_t tm = tm_create(segment_size, WORD);
  CHECK(tm != invalid_shared);
  void *first = tm_start(tm);

  auto field = [](void *segment, size_t index) {
    return at(segment, index * WORD);
  };
  auto next_of = [&](Tx &tx, void *segment) {
    return reinterpret_cast<void *>(tx.get(field(segment, NEXT)));
  };

  atomically(tm, false, [&](Tx &tx) {
    tx.set(field(first, COUNT), capacity);
    for (size_t i = 0; i < capacity; ++i)
      tx.set(field(first, BALANCES + i), initial);
  });

  // check the invariant, return the number of accounts
  auto audit = [&]() {
    size_t accounts;
    bool consistent;
    atomically(tm, true, [&](Tx &tx) {
      word_t sum = 0;
      accounts = 0;
      consistent = true;
      for (void *segment = first; segment; segment = next_of(tx, segment)) {
        size_t count = tx.get(field(segment, COUNT));
        sum += tx.get(field(segment, PARITY));
        for (size_t i = 0; i < count; ++i) {
          word_t balance = tx.get(field(segment, BALANCES + i));
          consistent &= balance >= 0;
          sum += balance;
        }
        accounts += count;
      }
      consistent &= sum == initial * static_cast<word_t>(accounts);
    });
    CHECK(consistent);
    return accounts;
  };

  // address of the balance of the index-th account, nullptr if none
  auto account = [&](Tx &tx, size_t index) -> void * {
    for (void *segment = first; segment; segment = next_of(tx, segment)) {
      size_t count = tx.get(field(segment, COUNT));
      if (index < count)
        return field(segment, BALANCES + index);
      index -= count;
    }
    return nullptr;
  };

  // move one unit between two accounts, return whether both exist
  auto transfer = [&](size_t from, size_t to) {
    bool found;
    atomically(tm, false, [&](Tx &tx) {
      void *source = account(tx, from);
      void *target = account(tx, to);
      found = source && target;
      if (!found)
        return;
      word_t balance = tx.get(source);
      if (balance > 0) {
        tx.set(source, balance - 1);
        tx.set(target, tx.get(target) + 1);
      }
    });
    return found;
  };

  // close the last account if there are more than 'limit', else open one
  auto resize = [&](size_t limit) {
    atomically(tm, false, [&](Tx &tx) {
      void *previous = nullptr, *last = first;
      size_t total = 0, count;
      while (true) {
        count = tx.get(field(last, COUNT));
        total += count;
        void *next = next_of(tx, last);
        if (!next)
          break;
        previous = last;
        last = next;
      }
      if (total > limit && total > 2) {
        word_t parity = tx.get(field(last, PARITY)) +
                        tx.get(field(last, BALANCES + count - 1)) - initial;
        if (count > 1) {
          tx.set(field(last, COUNT), count - 1);
          tx.set(field(last, PARITY), parity);
        } else { // the last segment gets empty: unlink and free it
          word_t previous_parity = tx.get(field(previous, PARITY));
          tx.set(field(previous, PARITY), previous_parity + parity);
          tx.set(field(previous, NEXT), 0);
          tx.free(last);
        }
      } else if (count < capacity) {
        tx.set(field(last, BALANCES + count), initial);
        tx.set(field(last, COUNT), count + 1);
      } else { // the last segment is full: link a new one
        void *segment = tx.alloc(segment_size);
        tx.set(field(segment, COUNT), 1);
        tx.set(field(segment, BALANCES), initial);
        tx.set(field(last, NEXT), reinterpret_cast<word_t>(segment));
      }
    });
  };

  parallel(threads, [&](size_t index) {
    std::minstd_rand rng(index + 1);
    std::bernoulli_distribution is_audit(0.5), is_resize(0.02);
    std::gamma_distribution<float> limit(target, 1);
    size_t accounts = capacity; // refreshed by each audit
    for (size_t i = 0; i < tx_per_thread; ++i) {
      if (is_audit(rng)) {
        accounts = audit();
      } else if (is_resize(rng)) {
        resize(limit(rng));
      } else {
        std::uniform_int_distribution<size_t> pick(0, accounts - 1);
        while (!transfer(pick(rng), pick(rng)))
          ;
      }
    }
  });

  audit();
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
  run("overwrite", test_overwrite);
  run("multiple regions", test_multiple_regions);
  run("many segments", test_many_segments);
  run("large segments", test_large_segments);
  run("counter", test_counter);
  run("swap", test_swap);
  run("bank", test_bank);
  return 0;
}
