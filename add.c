#include "rt.h"
#include <CL/opencl.h>
#include <assert.h>
#include <windows.h>
#include "ocl.h"

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

enum { N = 256 * 256 }; // Intel GPU limitation

static cl_int x_add_y(ocl_context_t* c, ocl_queue_t q, ocl_kernel_t k,
                      ocl_memory_t mx, ocl_memory_t my, ocl_memory_t mz) {
    {   // initialize pinned memory:
        float* x = ocl.map(q, ocl_map_write, mx, 0, N * sizeof(float));
        float* y = ocl.map(q, ocl_map_write, my, 0, N * sizeof(float));
        // two input vectors
        for (int32_t i = 0; i < N; i++) { x[i] = (float)i; y[i] = (float)(N - i); }
        ocl.unmap(q, mx, x);
        ocl.unmap(q, my, y);
    }
    ocl_arg_t args[] =
        {{&mx, sizeof(ocl_memory_t)},
         {&my, sizeof(ocl_memory_t)},
         {&mz, sizeof(ocl_memory_t)}
    };
    int64_t max_groups = ocl.devices[c->device_index].max_groups;
    int64_t max_items  = ocl.devices[c->device_index].max_items[0];
    int64_t groups = (N + max_items - 1) / max_items;
    int64_t items  = (N + groups - 1) / groups;
    assert(groups * items == N);
    ocl_event_t completion = ocl.enqueue_range_kernel(c, q, k, groups, items,
        countof(args), args);
    ocl.flush(q);
    ocl.finish(q);
    ocl.wait(&completion, 1);
    ocl_profiling_t p = {0};
    ocl.profile(completion, &p, N);
    ocl.dispose_event(completion);
    {   // map result and verify it:
        float* z = (float*)ocl.map(q, ocl_map_read, mz, 0, N * sizeof(float));
        for (int32_t i = 0; i < N; i++) {
            float xi = (float)i;
            float yi = (float)(N - i);
            assertion(xi + yi == z[i], "%.1f + %.1f = %.1f\n", xi, yi, z[i]);
        }
        ocl.unmap(q, mz, z);
    }
    traceln("kernel execution on device: %.3f us (microseconds) Gops=%.3f",
            p.time * (1000 * 1000), p.gops);
    return 0;
}

#define kernel_name "x_add_y"

static cl_int test(ocl_context_t* c) {
    cl_int result = 0;
    ocl_queue_t q = ocl.create_queue(c, true);
    static const char* code =
    "__kernel void " kernel_name "(__global const float* x, "
    "                              __global const float* y, "
    "                              __global float* z) {\n"
    "    int i = get_global_id(0);\n"
    "    z[i] = x[i] + y[i];\n"
    "}\n";
    ocl_program_t p = ocl.compile_program(c, code, strlen(code));
    ocl_kernel_t k = ocl.create_kernel(p, kernel_name);
    ocl_memory_t mx = ocl.allocate(c, ocl_allocate_write, N * sizeof(float));
    ocl_memory_t my = ocl.allocate(c, ocl_allocate_write, N * sizeof(float));
    ocl_memory_t mz = ocl.allocate(c, ocl_allocate_read, N * sizeof(float));
    x_add_y(c, q, k, mx, my, mz);
    x_add_y(c, q, k, mx, my, mz);
    x_add_y(c, q, k, mx, my, mz);
    ocl.deallocate(mx); // must be dealloca() before dispose_command_queue()
    ocl.deallocate(my);
    ocl.deallocate(mz);
    ocl.dispose_kernel(k);
    ocl.dispose_program(p);
    ocl.dispose_queue(q);
    return result;
}

int32_t main(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    cl_int result = 0;
    ocl.init();
    for (int cycles = 2; cycles > 0; cycles--) {
        for (int i = 0; i < ocl.device_count; i++) {
            ocl_context_t c = {0};
            ocl.open(&c, i);
            printf("%s\n", ocl.devices[i].name);
            result = test(&c);
            printf("test: %s\n", result == 0 ? "OK" : "FAILED");
            ocl.close(&c);
        }
    }
    return result;
}
