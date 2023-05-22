#include "rt.h"
#include <CL/opencl.h>
#include <assert.h>
#include <windows.h>
#include "ocl.h"

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

enum { N = 256 * 256 }; // Intel GPU limitation

// host side arrays (compare time)
static float X[N];
static float Y[N];
static float Z[N];

static double avg_time;
static double avg_user;
static double avg_host;
static double avg_gflops;

static cl_int x_add_y(ocl_context_t* c, ocl_kernel_t k,
                      ocl_memory_t mx, ocl_memory_t my, ocl_memory_t mz) {
    {   // initialize pinned memory:
        float* x = ocl.map(c, ocl_map_write, mx, 0, N * sizeof(float));
        float* y = ocl.map(c, ocl_map_write, my, 0, N * sizeof(float));
        // two input vectors
        for (int32_t i = 0; i < N; i++) { x[i] = (float)i; y[i] = (float)(N - i); }
        for (int32_t i = 0; i < N; i++) { X[i] = x[i]; Y[i] = y[i]; }
        ocl.unmap(c, mx, x);
        ocl.unmap(c, my, y);
    }
    ocl_arg_t args[] =
        {{&mx, sizeof(ocl_memory_t)},
         {&my, sizeof(ocl_memory_t)},
         {&mz, sizeof(ocl_memory_t)}
    };
//  int64_t max_groups = ocl.devices[c->ix].max_groups;
    int64_t max_items  = ocl.devices[c->ix].max_items[0];
    int64_t groups = (N + max_items - 1) / max_items;
    int64_t items  = (N + groups - 1) / groups;
    assert(groups * items == N);
    if (ocl.is_profiling(c)) { c->ov->profiling_count = 0; }
    double time = seconds();
    ocl_event_t done = ocl.enqueue_range_kernel(c, k, groups, items,
        countof(args), args);
    ocl_profiling_t* p = ocl.is_profiling(c) ? ocl.profile_add(c, done) : null;
    // flash() and finish() are unnecessary because ocl.wait(done)
//  ocl.flush(c);
//  ocl.finish(c);
    ocl.wait(&done, 1);
    time = seconds() - time;
    if (p != null) {
        p->user = time;
        p->count = N; // N kernel invocations
        p->fops = 1; //  1 floating operation each
        ocl.profile(p); // collect profiling info and calculate derived values
    }
    ocl.dispose_event(done); // client's responsibility
    float* z = (float*)ocl.map(c, ocl_map_read, mz, 0, N * sizeof(float));
    double host = 0;
    if (p != null) {
        // measure the same addition of N numbers on CPU
        enum { GB = 1024 * 1024 * 1024 }; // erase all L1, L2 and L3 caches
        byte_t* flash_caches = (byte_t*)malloc(1 * GB);
        for (int i = 0; i < 1 * GB; i++) { flash_caches[i] = (byte_t)i; }
        host = seconds();
        for (int32_t i = 0; i < N; i++) { Z[i] = X[i] + Y[i]; }
        host = seconds() - host;
        // prevent compiler from optimizing away the above loop
        for (int32_t i = 0; i < N; i++) {
            fatal_if(Z[i] != z[i], "%.1f + %.1f = %.1f\n", X[i], Y[i], z[i]);
        }
        free(flash_caches);
    } else { // just verify result:
        for (int32_t i = 0; i < N; i++) {
            fatal_if(X[i] + Y[i] != z[i], "%.1f + %.1f = %.1f instead of %.1f\n",
                X[i], Y[i], z[i], Z[i]);
        }
    }
    ocl.unmap(c, mz, z);
    if (p != null) {
        traceln("kernel: %6.3f user: %8.3f host: %7.3f (microsec) GFlops: %6.3f",
                p->time * USEC_IN_SEC, p->user * USEC_IN_SEC, host * USEC_IN_SEC,
                p->gflops);
        avg_time += p->time;
        avg_user += p->user;
        avg_host += host;
        avg_gflops += p->gflops;
    }
    return 0;
}

#define kernel_name "x_add_y"

static cl_int test(ocl_context_t* c) {
    cl_int result = 0;
    static const char* code =
    "__kernel void " kernel_name "(__global const float* x, "
    "                              __global const float* y, "
    "                              __global float* z) {\n"
    "    int i = get_global_id(0);\n"
    "    z[i] = x[i] + y[i];\n"
    "}\n";
    ocl_program_t p = ocl.compile_program(c, code, strlen(code), null);
    ocl_kernel_t k = ocl.create_kernel(p, kernel_name);
    ocl_memory_t mx = ocl.allocate(c, ocl_allocate_write, N * sizeof(float));
    ocl_memory_t my = ocl.allocate(c, ocl_allocate_write, N * sizeof(float));
    ocl_memory_t mz = ocl.allocate(c, ocl_allocate_read, N * sizeof(float));
    enum { M = 16 }; // measurements
    for (int i = 0; i < M; i++) {
        x_add_y(c, k, mx, my, mz);
    }
    if (ocl.is_profiling(c)) {
        avg_time /= M;
        avg_user /= M;
        avg_host /= M;
        avg_gflops /= M;
        traceln("average");
        traceln("kernel: %6.3f user: %8.3f host: %7.3f (microsec) GFlops: %6.3f",
                avg_time * USEC_IN_SEC, avg_user * USEC_IN_SEC, avg_host * USEC_IN_SEC,
                avg_gflops);
    }
    // NVIDIA GeForce RTX 3080 Laptop GPU
    // kernel:  6.330 user:  193.594 host:  57.825 (microsec) GFlops: 10.412
    // Intel(R) UHD Graphics
    // kernel: 27.345 user:  865.303 host:  56.305 (microsec) GFlops:  3.051
    ocl.deallocate(mx); // must be dealloca() before dispose_command_queue()
    ocl.deallocate(my);
    ocl.deallocate(mz);
    ocl.dispose_kernel(k);
    ocl.dispose_program(p);
    return result;
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    cl_int result = 0;
    ocl.init();
    for (int cycles = 2; cycles > 0; cycles--) {
        for (int i = 0; i < ocl.count; i++) {
            ocl_context_t c = ocl.open(i, null);
            traceln("%s\n", ocl.devices[i].name);
            result = test(&c);
            traceln("test: %s\n", result == 0 ? "OK" : "FAILED");
            ocl.close(&c);
        }
    }
    ocl_profiling_t p[4096];
    ocl_override_t ov = {
        .max_groups = 0,
        .max_items = 0,
        .profiling = p,
        .max_profiling_count = countof(p),
        .profiling_count = 0
    };
    // profiling measurement:
    for (int i = 0; i < ocl.count; i++) {
        ocl_context_t c = ocl.open(i, &ov);
        traceln("%s", ocl.devices[i].name);
        result = test(&c);
        traceln("test: %s\n", result == 0 ? "OK" : "FAILED");
        ocl.close(&c);
    }
    return result;
}
