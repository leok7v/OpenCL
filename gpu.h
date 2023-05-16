#pragma once
#include "ocl.h"

#ifndef fp16_t
    #define fp16_t half // ???
#endif

#ifndef fp32_t
    #define fp32_t float
#endif

#ifndef fp64_t
    #define fp64_t double
#endif

enum { // .allocate() flags (matching OpenCL)
    gpu_allocate_read  = (1 << 2),
    gpu_allocate_write = (1 << 1),
    gpu_allocate_rw    = (1 << 0)
};

enum { // kernel indices
    gpu_fp16 = 0,
    gpu_fp32 = 1,
    gpu_fp64 = 2
};

// TODO: gpu_t could be opaque union w/o direct dependency on ocl.h

typedef struct gpu_s {
    ocl_context_t c;
    // kernels are properties of c.c ocl_context:
    ocl_kernel_t dot[3];
    ocl_kernel_t sum_odd[3];
    ocl_kernel_t sum_even[3];
} gpu_t;

typedef struct gpu_memory_s { // treat as read only, will change don't cache
    void*  m; // mapped memory address in virtual memory.
    void*  handle;
    size_t bytes;
    gpu_t* gpu;
    uint32_t map; // mapping flags
} gpu_memory_t;

typedef struct gpu_if {
    void (*init)(gpu_t* gp, ocl_context_t* c, const char* code, int32_t bytes);
    // memory pointers returned by allocate() are "8 bytes alligned  "locked"
    // in the host memory and can be accessed by the host directly.
    gpu_memory_t (*allocate)(gpu_t* gpu, int flags, size_t bytes);
    void (*deallocate)(gpu_memory_t* gm);
    // Only allocated GPU shared memory can be used as arguments.
    // functions will unmap host memory and allow access by the GPU
    // and will remap it back when done with a hope address won't change.
    //
    // TODO: investigate if we unmapping and remapping does not move
    // the memory in virtual address space.
    // And if addresses of remapping will change we will need to rethink
    // the whole scheme and pass arguments in the form fp32_t *v[] instead
    // and have it modified on remapping.
    fp32_t (*dot_f32)( // offsets and strides are in elemnts counts
        gpu_memory_t* v0, int64_t offset0, int64_t stride0,
        gpu_memory_t* v1, int64_t offset1, int64_t stride1,
        int64_t n);
    void (*gemv_f32)(gpu_memory_t* vec, gpu_memory_t* matrixx,
                 gpu_memory_t* result,
                 int64_t n, int64_t m, // vec[n], matrix[m][n], result[m]
                 int64_t stride_v, int64_t stride_mx[2]);
    void (*fini)(gpu_t* gp);
} gpu_if;

extern gpu_if gpu;

#ifdef TODO // (on "as needed" basis)
    Level 1 BLAS (14 subprograms):
    [ ] asum
        axpy
        copy
    [x] dot
        iamax
        nrm2
        rot
        rotg
        rotm
        rotmg
        scal
        swap
        sdsdot
        dsdot

    Level 2 BLAS (6 subprograms):
    [ ] gemv
        gbmv
        hemv
        hbmv
        symv
        sbmv

    Level 3 BLAS (4 subprograms)
        gemm
        symm
        hemm
        syrk
        herk
        syr2k
        her2k
        trmm
        trsm
#endif