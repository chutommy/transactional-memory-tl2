# TL2 software transactional memory

A word-based software transactional memory in C++17, implementing [Transactional Locking II](https://doi.org/10.1007/11864219_14) (Dice, Shalev, Shavit, 2006).

STM allows threads to share a memory region and access it inside transactions: a transaction either commits all its writes at once, or aborts and leaves no trace.
There is no global lock: transactions run optimistically and only conflict when they touch the same words.

## Usage

```cpp
#include "tm.hpp"

shared_t tm = tm_create(64, 8); // one segment of 64 bytes, in words of 8 bytes
void *counter = tm_start(tm);   // opaque address of its first word

// increment the counter, retrying until the transaction commits
for (bool committed = false; !committed;) {

  tx_t tx = tm_begin(tm, false);

  // read
  uint64_t value;
  if (!tm_read(tm, tx, counter, 8, &value))
    continue; // abort

  // increment
  value += 1;

  // write
  if (!tm_write(tm, tx, &value, 8, counter))
    continue; // abort

  committed = tm_end(tm, tx);
}

tm_destroy(tm);
```

`tm_alloc` and `tm_free` add and remove segments inside a transaction.
See `tm.hpp` for the full interface.

## Build and test

```sh
make       # build/libtm.a
make test  # build and run the test suite
```

Requires a C++17 compiler and `make`. Developed for x86-64 Linux, also builds on arm64 macOS.

## How it works

See <https://chutommy.com/projects/transactional-memory-tl2/>.
