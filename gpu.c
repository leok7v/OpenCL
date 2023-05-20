#include "gpu.h"
#include "rt.h"
#include <malloc.h>
#include <string.h>

static_assertion(gpu_allocate_read  == 0);
static_assertion(gpu_allocate_write == 1);
static_assertion(gpu_allocate_rw    == 2);

static int gpu_allocate_to_ocl[] = {
    ocl_allocate_read,
    ocl_allocate_write,
    ocl_allocate_rw
};

static_assertion(gpu_map_read  == 0);
static_assertion(gpu_map_write == 1);
static_assertion(gpu_map_rw    == 2);

static int gpu_map_to_ocl[] = {
    ocl_map_read,
    ocl_map_write,
    ocl_map_rw
};

static gpu_memory_t gpu_allocate(gpu_t* g, int access, int64_t bytes) {
    gpu_memory_t gm;
    gm.gpu = g;
    gm.bytes = bytes;
    gm.handle = ocl.allocate(&g->c, gpu_allocate_to_ocl[access], bytes);
//  traceln("%p: %p", gm->handle, gm->m);
    return gm;
}

static void gpu_deallocate(gpu_memory_t* gm) {
//  traceln("%p: %p", gm->handle, gm->m);
    ocl.deallocate((ocl_memory_t)gm->handle);
    memset(gm, 0, sizeof(gm));
}

static void* gpu_map(gpu_memory_t* gm, int mapping, int64_t offset, int64_t bytes) {
    gm->m = ocl.map(&gm->gpu->c, gpu_map_to_ocl[mapping],
        (ocl_memory_t)gm->handle, offset, bytes);
//  traceln("%p: %p", gm->handle, gm->m);
    return gm->m;
}

static void gpu_unmap(gpu_memory_t* gm) {
//  traceln("%p: %p", gm->handle, gm->m);
    ocl.unmap(&gm->gpu->c, (ocl_memory_t)gm->handle, gm->m);
    gm->m = null;
}

static void gpu_init(gpu_t* g, ocl_context_t* c) {
    g->c = *c;
}

static void gpu_fini(gpu_t* g) {

// TODO: ??? why copy not reference?
//  g->c = null;
}

gpu_if gpu = {
    .init       = gpu_init,
    .allocate   = gpu_allocate,
    .deallocate = gpu_deallocate,
    .map        = gpu_map,
    .unmap      = gpu_unmap,
    .fini       = gpu_fini
};

