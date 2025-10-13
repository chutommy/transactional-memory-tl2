/**
 * @file   tm.hpp
 * @author Tommy Chu <tommychu2256@gmail.com>
 *
 * @section DESCRIPTION
 *
 * Interface of the software transactional memory.
 *
 * A shared memory region is a set of segments, accessed in words of 'align'
 * bytes. Addresses inside a region are opaque: they are only meaningful to
 * tm_read, tm_write and tm_free, and must not be dereferenced.
 *
 * A thread runs at most one transaction at a time. When an operation reports
 * an abort (tm_read, tm_write or tm_free returning false, tm_alloc returning
 * Alloc::abort), the transaction is over: tm_end must not be called and the
 * caller is expected to retry from tm_begin.
 **/

#pragma once

#include <cstddef>
#include <cstdint>

/** Opaque handle of a shared memory region.
 **/
using shared_t = void *;
constexpr shared_t invalid_shared = nullptr;

/** Opaque handle of a running transaction.
 **/
using tx_t = uintptr_t;
constexpr tx_t invalid_tx = ~tx_t(0);

/** Outcome of tm_alloc.
 **/
enum class Alloc : int {
  success = 0, // segment allocated, the transaction can continue
  abort = 1,   // the transaction aborted and can be retried
  nomem = 2    // out of memory, the transaction can continue
};

extern "C" {

/** Create a region with a first, zeroed, non-freeable segment of 'size' bytes.
 * 'align' is a power of 2 and 'size' a positive multiple of it.
 * @return Region handle, 'invalid_shared' on failure
 **/
shared_t tm_create(size_t size, size_t align) noexcept;

/** Destroy a region on which no transaction is running.
 **/
void tm_destroy(shared_t shared) noexcept;

/** Address, size (in bytes) and alignment (in bytes) of the first segment.
 **/
void *tm_start(shared_t shared) noexcept;
size_t tm_size(shared_t shared) noexcept;
size_t tm_align(shared_t shared) noexcept;

/** Begin a transaction, read-only if 'is_ro'.
 * @return Transaction handle, 'invalid_tx' on failure
 **/
tx_t tm_begin(shared_t shared, bool is_ro) noexcept;

/** Commit a transaction.
 * @return Whether the transaction committed (false: aborted, can be retried)
 **/
bool tm_end(shared_t shared, tx_t tx) noexcept;

/** Copy 'size' bytes from the shared 'source' to the private 'target'.
 * 'size' is a positive multiple of the alignment.
 * @return Whether the transaction can continue (false: aborted)
 **/
bool tm_read(shared_t shared, tx_t tx, void const *source, size_t size,
             void *target) noexcept;

/** Copy 'size' bytes from the private 'source' to the shared 'target'.
 * 'size' is a positive multiple of the alignment.
 * @return Whether the transaction can continue (false: aborted)
 **/
bool tm_write(shared_t shared, tx_t tx, void const *source, size_t size,
              void *target) noexcept;

/** Allocate a zeroed segment of 'size' bytes (a positive multiple of the
 * alignment) and store its address in 'target'.
 **/
Alloc tm_alloc(shared_t shared, tx_t tx, size_t size, void **target) noexcept;

/** Free a segment obtained from tm_alloc, once the transaction commits.
 * @return Whether the transaction can continue (false: aborted)
 **/
bool tm_free(shared_t shared, tx_t tx, void *target) noexcept;

} // extern "C"
