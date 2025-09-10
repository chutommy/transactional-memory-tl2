/**
 * @file   tm.hpp
 * @author Tommy Chu <tommychu2256@gmail.com>
 *
 * @section DESCRIPTION
 *
 * Interface of the software transactional memory.
 **/

#pragma once

#include <cstddef>
#include <cstdint>

using shared_t = void *;
constexpr shared_t invalid_shared = nullptr;

using tx_t = uintptr_t;
constexpr tx_t invalid_tx = ~tx_t(0);

enum class Alloc : int {
  success = 0, // segment allocated
  abort = 1,   // transaction aborted
  nomem = 2    // out of memory
};

extern "C" {

// region: first segment of 'size' bytes, words of 'align' bytes
shared_t tm_create(size_t size, size_t align) noexcept;
void tm_destroy(shared_t shared) noexcept;
void *tm_start(shared_t shared) noexcept;
size_t tm_size(shared_t shared) noexcept;
size_t tm_align(shared_t shared) noexcept;

// transaction: false means aborted, the transaction is over
tx_t tm_begin(shared_t shared, bool is_ro) noexcept;
bool tm_end(shared_t shared, tx_t tx) noexcept;
bool tm_read(shared_t shared, tx_t tx, void const *source, size_t size,
             void *target) noexcept;
bool tm_write(shared_t shared, tx_t tx, void const *source, size_t size,
              void *target) noexcept;
Alloc tm_alloc(shared_t shared, tx_t tx, size_t size, void **target) noexcept;
bool tm_free(shared_t shared, tx_t tx, void *target) noexcept;

} // extern "C"
