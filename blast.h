#pragma once
#include "ocl.h" // TODO: this is temporarely, needs to be hidden in implementation
#include "fp16.h"
#include "gpu.h"

typedef struct blast_s blast_t;

typedef struct blast_s {
    gpu_t* gpu;
    // Only the memory allocated by gpu.allocate() can be used as an arguments.
    // Caller MUST unmap that memory to allow access to it by the GPU.
    // and will remap it back when done. The address WILL CHANGE!
    // The offset parameters are necessary because multiple vectors and
    // matrices can be kept inside one gpu.allocated() memory region.
    //
    // The functions pointer can be null if fp16 or fp64 is not supported:
    // sum()
    fp64_t (*sum_fp16)(blast_t* b,
        gpu_memory_t* v0, int64_t offset0, int64_t stride0,int64_t n);
    fp64_t (*sum_fp32)(blast_t* b,
        gpu_memory_t* v0, int64_t offset0, int64_t stride0,int64_t n);
    fp64_t (*sum_fp64)(blast_t* b,
        gpu_memory_t* v0, int64_t offset0, int64_t stride0,int64_t n);
    // dot()
    fp64_t (*dot_fp16)(blast_t* b,
        gpu_memory_t* v0, int64_t offset0, int64_t stride0,
        gpu_memory_t* v1, int64_t offset1, int64_t stride1, int64_t n);
    fp64_t (*dot_fp32)(blast_t* b,
        gpu_memory_t* v0, int64_t offset0, int64_t stride0,
        gpu_memory_t* v1, int64_t offset1, int64_t stride1, int64_t n);
    fp64_t (*dot_fp64)(blast_t* b,
        gpu_memory_t* v0, int64_t offset0, int64_t stride0,
        gpu_memory_t* v1, int64_t offset1, int64_t stride1, int64_t n);
    // gemv()
    void (*gemv_f16)(blast_t* b,
        gpu_memory_t* matrix/*[m][n]*/, int64_t offset_m, int64_t stride_m,
        gpu_memory_t* vector/*[n]*/,    int64_t offset_v, int64_t stride_v,
        gpu_memory_t* result/*[m]*/, int64_t m, int64_t n);
    void (*gemv_f32)(blast_t* b,
        gpu_memory_t* matrix/*[m][n]*/, int64_t offset_m, int64_t stride_m,
        gpu_memory_t* vector/*[n]*/,    int64_t offset_v, int64_t stride_v,
        gpu_memory_t* result/*[m]*/, int64_t m, int64_t n);
    void (*gemv_f64)(blast_t* b,
        gpu_memory_t* matrix/*[m][n]*/, int64_t offset_m, int64_t stride_m,
        gpu_memory_t* vector/*[n]*/,    int64_t offset_v, int64_t stride_v,
        gpu_memory_t* result/*[m]*/, int64_t m, int64_t n);
    // kernels are properties of c.c ocl_context:
    ocl_kernel_t dot[3];     // gpu_fp16, gpu_fp32, gpu_fp64
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
} blast_t;

typedef struct blast_if {
    void (*init)(blast_t* b, gpu_t* g);
    void (*fini)(blast_t* b);
} blast_if;

extern blast_if blast;

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