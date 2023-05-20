#pragma once
#include "ocl.h" // TODO: this is temporarely, needs to be hidden in implementation
#include "fp16.h"

enum { // .allocate() flags (for now matching ocl_allocate_*)
    gpu_allocate_read  = 0, // not a bitset!
    gpu_allocate_write = 1,
    gpu_allocate_rw    = 2
};

enum { // .map() access flags (for now matching ocl_map_*)
    gpu_map_read  = 0, // not a bitset!
    gpu_map_write = 1, // invalidates region
    gpu_map_rw    = 2,
};

enum { // kernel indices
    gpu_fp16 = 0,
    gpu_fp32 = 1,
    gpu_fp64 = 2
};

// TODO: gpu_t could be opaque union w/o direct dependency on ocl.h

typedef struct gpu_s {
    ocl_context_t c;
} gpu_t;

typedef struct gpu_memory_s { // treat as read only, will change don't cache
    void*  m; // mapped memory address in virtual memory.
    void*  handle;
    size_t bytes;
    gpu_t* gpu;
} gpu_memory_t;

typedef struct gpu_if {
    void (*init)(gpu_t* gp, ocl_context_t* c);
    // memory pointers returned by allocate() are "8 bytes alligned  "locked"
    // in the host memory and can be accessed by the host directly.
    gpu_memory_t (*allocate)(gpu_t* gpu, int access, int64_t bytes);
    void  (*deallocate)(gpu_memory_t* gm);
    // Client must map gm to host memory to access it and unmap before
    // invocation of any other BLAS operation below
    void* (*map)(gpu_memory_t* gm, int mapping, int64_t offset, int64_t bytes);
    void  (*unmap)(gpu_memory_t* gm);
    void (*fini)(gpu_t* gp);
} gpu_if;

extern gpu_if gpu;

