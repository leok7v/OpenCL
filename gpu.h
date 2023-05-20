#pragma once
#include "ocl.h" // TODO: this is temporarely, needs to be hidden in implementation
#include "fp16.h"

#ifndef fp32_t
    #define fp32_t float
#endif

#ifndef fp64_t
    #define fp64_t double
#endif

enum { // .allocate() flags (for now matching ocl_allocate_*)
    gpu_allocate_read  = (1 << 2),
    gpu_allocate_write = (1 << 1),
    gpu_allocate_rw    = (1 << 0)
};

enum { // .map() access flags (for now matching ocl_map_*)
    gpu_map_read  = (1 << 0),
    gpu_map_write = (1 << 2), // invalidates region
    gpu_map_rw    = ((1 << 0) | (1 << 1)),
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
    ocl_kernel_t dot[3];     // fp16, fp32, fp64
    ocl_kernel_t dot_os[3];  // offset + stride
    ocl_kernel_t sum_odd[3];
    ocl_kernel_t sum_odd_os[3];
    ocl_kernel_t sum_even[3];
    ocl_kernel_t sum_even_os[3];
    ocl_kernel_t gemv[3];
    ocl_kernel_t gemv_os[3];
    // TODO:
    ocl_kernel_t fma[3];
    ocl_kernel_t fma_os[3];
    ocl_kernel_t mad[3];
    ocl_kernel_t mad_os[3];
} gpu_t;

typedef struct gpu_memory_s { // treat as read only, will change don't cache
    void*  m; // mapped memory address in virtual memory.
    void*  handle;
    size_t bytes;
    gpu_t* gpu;
} gpu_memory_t;

typedef struct gpu_if {
    void (*init)(gpu_t* gp, ocl_context_t* c, const char* code, int32_t bytes);
    // memory pointers returned by allocate() are "8 bytes alligned  "locked"
    // in the host memory and can be accessed by the host directly.
    gpu_memory_t (*allocate)(gpu_t* gpu, int access, int64_t bytes);
    void  (*deallocate)(gpu_memory_t* gm);
    // Client must map gm to host memory to access it and unmap before
    // invocation of any other BLAS operation below
    void* (*map)(gpu_memory_t* gm, int mapping, int64_t offset, int64_t bytes);
    void  (*unmap)(gpu_memory_t* gm);
    // Only the memory allocated by gpu.allocate() can be used as an arguments.
    // Caller MUST unmap that memory to allow access to it by the GPU.
    // and will remap it back when done. The address WILL CHANGE!
    // The offset parameters are necessary because multiple vectors and
    // matrices can be kept inside one gpu.allocated() memory region.
    fp16_t (*dot_fp16)(gpu_memory_t* v0, int64_t offset0, int64_t stride0,
                       gpu_memory_t* v1, int64_t offset1, int64_t stride1,
                       int64_t n);
    fp32_t (*dot_fp32)(gpu_memory_t* v0, int64_t offset0, int64_t stride0,
                       gpu_memory_t* v1, int64_t offset1, int64_t stride1,
                       int64_t n);
    fp64_t (*dot_fp64)(gpu_memory_t* v0, int64_t offset0, int64_t stride0,
                       gpu_memory_t* v1, int64_t offset1, int64_t stride1,
                       int64_t n);
    void (*gemv_f32)(
        gpu_memory_t* vector/*[n]*/,    int64_t offset_v, int64_t stride_v,
        gpu_memory_t* matrix/*[m][n]*/, int64_t offset_m, int64_t stride_m,
        gpu_memory_t* result/*[m]*/, int64_t m, int64_t n);
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