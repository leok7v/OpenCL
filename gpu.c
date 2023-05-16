#include "gpu.h"
#include "rt.h"
#include <malloc.h>
#include <string.h>

static_assertion(gpu_allocate_read  == ocl_allocate_read );
static_assertion(gpu_allocate_write == ocl_allocate_write);
static_assertion(gpu_allocate_rw    == ocl_allocate_rw);

static gpu_memory_t gpu_allocate(gpu_t* g, int flags, size_t bytes) {
    gpu_memory_t gm;
    gm.gpu = g;
    gm.bytes = ((bytes + 7) & ~7); // aligned to 8 bytes
    ocl_memory_t handle = ocl.allocate(&g->c, flags, gm.bytes);
    gm.map = 0; // map flags
    switch (flags) {
        case gpu_allocate_read:  gm.map = ocl_map_read;  break;
        case gpu_allocate_write: gm.map = ocl_map_write; break;
        case gpu_allocate_rw:    gm.map = ocl_map_rw;    break;
        default: fatal_if(true, "invalid flags %d", flags);
    }
    gm.m = ocl.map(&g->c, gm.map, handle, /*offset:*/ 0, gm.bytes);
    gm.handle = handle;
    return gm;
}

static void gpu_deallocate(gpu_memory_t* gm) {
    ocl.unmap(&gm->gpu->c, (ocl_memory_t)gm->handle, gm->m);
    ocl.deallocate((ocl_memory_t)gm->handle);
}

static ocl_program_t gpu_compile(gpu_t* g, const char* suffix,
        const void* code, int bytes) {
    const char* fp_t = null;
    if (strcmp(suffix, "fp32") == 0) {
        fp_t = "float";
    } else if (strcmp(suffix, "fp64") == 0) {
        fp_t = "double";
    } else if (strcmp(suffix, "fp16") == 0) {
        fp_t = "float16";
    } else {
        fatal_if(true, "unsupported suffix: %s", suffix);
    }
    int n = bytes + 1024;
    char* text = alloca(n);
    snprintf(text, n, "\n"
        "#define fp_t %s\n"
        "#define suffix %s\n"
        "%*.*s",
        fp_t,
        suffix, bytes, bytes, (const char*)code);
//  printf("%s\n", text);
    return ocl.compile_program(&g->c, text, strlen(text));
}

extern fp32_t gpu_dot_f32(
    gpu_memory_t* v0, int64_t offset0, int64_t stride0,
    gpu_memory_t* v1, int64_t offset1, int64_t stride1,
    int64_t n);

static void gpu_init(gpu_t* g, ocl_context_t* c,
        const char* code, int32_t bytes) {
    g->c = *c;
    static_assertion(gpu_fp16 == 0 && gpu_fp32 == 1 && gpu_fp64 == 2);
    ocl_program_t p[3] = {
        gpu_compile(g, "fp16", code, bytes),
        gpu_compile(g, "fp32", code, bytes),
        gpu_compile(g, "fp64", code, bytes)
    };
    const char* sum_odd[]  = {"sum_odd_fp16", "sum_odd_fp32", "sum_odd_fp64"};
    const char* sum_even[] = {"sum_even_fp16", "sum_even_fp32", "sum_even_fp64"};
    const char* dot[]      = {"dot_fp16", "dot_fp32", "dot_fp64"};
    for (int fp = gpu_fp16; fp <= gpu_fp64; fp++) {
        g->sum_odd [fp] = ocl.create_kernel(p[fp], sum_odd[fp]);
        g->sum_even[fp] = ocl.create_kernel(p[fp], sum_even[fp]);
        g->dot[fp]      = ocl.create_kernel(p[fp], dot[fp]);
        ocl.dispose_program(p[fp]);
    }
    gpu.dot_f32 = gpu_dot_f32;
}

static void gpu_fini(gpu_t* g) {
    for (int fp = gpu_fp16; fp <= gpu_fp64; fp++) {
        ocl.dispose_kernel(g->sum_odd [fp]);
        ocl.dispose_kernel(g->sum_even[fp]);
        ocl.dispose_kernel(g->dot[fp]);
    }
}

gpu_if gpu = {
    .init = gpu_init,
    .allocate = gpu_allocate,
    .deallocate = gpu_deallocate,
    .dot_f32 = gpu_dot_f32,
    .fini = gpu_fini
};

