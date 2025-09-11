/**
 * @file   tm.cpp
 * @author Tommy Chu <tommychu2256@gmail.com>
 *
 * @section DESCRIPTION
 *
 * Reference implementation: one global lock per region, held for the whole
 * transaction. Nothing ever aborts, and nothing ever runs concurrently either,
 * this is only a baseline to write the tests against.
 **/

#define _POSIX_C_SOURCE 200809L

#include "tm.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

struct Region {
  std::mutex lock;
  void *start;
  size_t size;
  size_t align;
  std::vector<void *> segments; // allocated by tm_alloc, not freed yet
};

static void *zeroed(size_t size, size_t align) {
  void *segment;
  if (posix_memalign(&segment, std::max(align, sizeof(void *)), size) != 0)
    return nullptr;
  std::memset(segment, 0, size);
  return segment;
}

shared_t tm_create(size_t size, size_t align) noexcept {
  Region *region = new (std::nothrow) Region;
  if (!region)
    return invalid_shared;
  region->start = zeroed(size, align);
  if (!region->start) {
    delete region;
    return invalid_shared;
  }
  region->size = size;
  region->align = align;
  return region;
}

void tm_destroy(shared_t shared) noexcept {
  Region *region = static_cast<Region *>(shared);
  for (void *segment : region->segments)
    free(segment);
  free(region->start);
  delete region;
}

void *tm_start(shared_t shared) noexcept {
  return static_cast<Region *>(shared)->start;
}

size_t tm_size(shared_t shared) noexcept {
  return static_cast<Region *>(shared)->size;
}

size_t tm_align(shared_t shared) noexcept {
  return static_cast<Region *>(shared)->align;
}

tx_t tm_begin(shared_t shared, bool /* is_ro */) noexcept {
  static_cast<Region *>(shared)->lock.lock();
  return 0; // anything but invalid_tx, there is no context
}

bool tm_end(shared_t shared, tx_t /* tx */) noexcept {
  static_cast<Region *>(shared)->lock.unlock();
  return true;
}

bool tm_read(shared_t /* shared */, tx_t /* tx */, void const *source,
             size_t size, void *target) noexcept {
  std::memcpy(target, source, size);
  return true;
}

bool tm_write(shared_t /* shared */, tx_t /* tx */, void const *source,
              size_t size, void *target) noexcept {
  std::memcpy(target, source, size);
  return true;
}

Alloc tm_alloc(shared_t shared, tx_t /* tx */, size_t size,
               void **target) noexcept {
  Region *region = static_cast<Region *>(shared);
  void *segment = zeroed(size, region->align);
  if (!segment)
    return Alloc::nomem;
  try {
    region->segments.push_back(segment);
  } catch (...) {
    free(segment);
    return Alloc::nomem;
  }
  *target = segment;
  return Alloc::success;
}

// nothing aborts and no one else runs: the segment can go right away
bool tm_free(shared_t shared, tx_t /* tx */, void *target) noexcept {
  Region *region = static_cast<Region *>(shared);
  auto it = std::find(region->segments.begin(), region->segments.end(), target);
  region->segments.erase(it);
  free(target);
  return true;
}
