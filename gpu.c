#include "gpu.h"
#include "rt.h"
#include <malloc.h>
#include <string.h>

static_assertion(gpu_allocate_read  == ocl_allocate_read );
static_assertion(gpu_allocate_write == ocl_allocate_write);
static_assertion(gpu_allocate_rw    == ocl_allocate_rw);

static_assertion(gpu_map_read  == ocl_map_read );
static_assertion(gpu_map_write == ocl_map_write);
static_assertion(gpu_map_rw    == ocl_map_rw);

static gpu_memory_t gpu_allocate(gpu_t* g, int access, int64_t bytes) {
    gpu_memory_t gm;
    gm.gpu = g;
    gm.bytes = bytes;
    gm.handle = ocl.allocate(&g->c, access, bytes);
//  traceln("%p: %p", gm->handle, gm->m);
    return gm;
}

static void gpu_deallocate(gpu_memory_t* gm) {
//  traceln("%p: %p", gm->handle, gm->m);
    ocl.deallocate((ocl_memory_t)gm->handle);
    memset(gm, 0, sizeof(gm));
}

static void* gpu_map(gpu_memory_t* gm, int mapping, int64_t offset, int64_t bytes) {
    gm->m = ocl.map(&gm->gpu->c, mapping,
        (ocl_memory_t)gm->handle, offset, bytes);
//  traceln("%p: %p", gm->handle, gm->m);
    return gm->m;
}

static void gpu_unmap(gpu_memory_t* gm) {
//  traceln("%p: %p", gm->handle, gm->m);
    ocl.unmap(&gm->gpu->c, (ocl_memory_t)gm->handle, gm->m);
    gm->m = null;
}

static const char* gpu_program_options(gpu_t* g, int kind) {
    static const char* type_t[] = {"half", "float", "double"};
    static const char* suffix[] = {"fp16", "fp32", "fp64"};
    const char* fp_t = type_t[kind];
    // see https://man.opencl.org/clBuildProgram.html
    const ocl_device_t* d = &ocl.devices[g->c.ix];
    static char options[4096];
    char* p = options;
    #pragma push_macro("append")
    #define append(format, ...) do {                                     \
        intptr_t k = options + countof(options) - p - 1;                 \
        fatal_if(k <= 0, "options[%d] overflow", (int)countof(options)); \
        snprintf(p, k, format, __VA_ARGS__);                             \
        p += k;                                                          \
    } while (0)
    append("-cl-std=CL%d.%d "
           "-D fp_t=%s -D vec4=%s4 -D vec8=%s8 -D vec16=%s16 -D suffix=%s %s ",
           d->c_version_major, d->c_version_minor,
           fp_t, fp_t,fp_t, fp_t, suffix[kind],
          (kind == gpu_fp16 ? "-D fp16_surrogate" : ""));
    #pragma pop_macro("append")
    *p = 0;
    traceln("options: %s", options);
    return options;
}

static ocl_program_t gpu_compile(gpu_t* g, int kind,
        const void* code, int bytes) {
    traceln("kind: %d", kind);
//  traceln("%*.*s\n\n", bytes, bytes, code);
    return ocl.compile_program(&g->c, code, bytes,
        gpu_program_options(g, kind));
}

extern void dot_init(); // see: dot.c

static void gpu_init(gpu_t* g, ocl_context_t* c,
        const char* code, int32_t bytes) {
    g->c = *c;
    static_assertion(gpu_fp16 == 0 && gpu_fp32 == 1 && gpu_fp64 == 2);
    ocl_program_t p[3] = {
        gpu_compile(g, gpu_fp16, code, bytes),
        gpu_compile(g, gpu_fp32, code, bytes),
        gpu_compile(g, gpu_fp64, code, bytes)
    };
    static const char* sum_odd[]     = {"sum_odd_fp16",     "sum_odd_fp32",     "sum_odd_fp64"};
    static const char* sum_odd_os[]  = {"sum_odd_os_fp16",  "sum_odd_os_fp32",  "sum_odd_os_fp64"};
    static const char* sum_even[]    = {"sum_even_fp16",    "sum_even_fp32",    "sum_even_fp64"};
    static const char* sum_even_os[] = {"sum_even_os_fp16", "sum_even_os_fp32", "sum_even_os_fp64"};
    static const char* dot[]         = {"dot_fp16",         "dot_fp32",         "dot_fp64"};
    static const char* dot_os[]      = {"dot_os_fp16",      "dot_os_fp32",      "dot_os_fp64"};
    static const char* gemv[]        = {"gemv_fp16",        "gemv_fp32",        "gemv_fp64"};
    static const char* gemv_os[]     = {"gemv_os_fp16",     "gemv_os_fp32",     "gemv_os_fp64"};
    for (int fp = gpu_fp16; fp <= gpu_fp64; fp++) {
        g->sum_odd[fp]     = ocl.create_kernel(p[fp], sum_odd[fp]);
        g->sum_odd_os[fp]  = ocl.create_kernel(p[fp], sum_odd_os[fp]);
        g->sum_even[fp]    = ocl.create_kernel(p[fp], sum_even[fp]);
        g->sum_even_os[fp] = ocl.create_kernel(p[fp], sum_even_os[fp]);
        g->dot[fp]         = ocl.create_kernel(p[fp], dot[fp]);
        g->dot_os[fp]      = ocl.create_kernel(p[fp], dot_os[fp]);
        g->gemv[fp]        = ocl.create_kernel(p[fp], gemv[fp]);
        g->gemv_os[fp]     = ocl.create_kernel(p[fp], gemv_os[fp]);
        ocl.dispose_program(p[fp]);
    }
    dot_init();
}

static void gpu_fini(gpu_t* g) {
    for (int fp = gpu_fp16; fp <= gpu_fp64; fp++) {
        ocl.dispose_kernel(g->sum_odd [fp]);
        ocl.dispose_kernel(g->sum_even[fp]);
        ocl.dispose_kernel(g->dot[fp]);
    }
}

gpu_if gpu = {
    .init       = gpu_init,
    .allocate   = gpu_allocate,
    .deallocate = gpu_deallocate,
    .map        = gpu_map,
    .unmap      = gpu_unmap,
    .fini       = gpu_fini
};

