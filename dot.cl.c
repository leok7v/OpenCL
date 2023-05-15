#include "rt.h"
#include <CL/opencl.h>
#include <assert.h>
#include <math.h>
#include <windows.h>
#include "ocl.h"

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

enum { N = 16 };

#define fp32_t float
#define fp64_t double


#define kernel(kernel_name, fp_t)                                         \
    "#pragma OPENCL EXTENSION cl_khr_fp64: enable                    \n"  \
    "#pragma OPENCL EXTENSION cl_khr_fp16: enable                    \n"  \
    "                                                                \n"  \
    "typedef struct { " fp_t " vec; } v4f_t;                         \n"  \
    "                                                                \n"  \
    "__kernel void " kernel_name fp_t "(                             \n"  \
    "        __global const v4f_t* x, __global const v4f_t* y,       \n"  \
    "        __global " fp_t "* z ,__local  " fp_t "* partial) {     \n"  \
    "   const int gid = get_global_id(0);                            \n"  \
    "   const int lid = get_local_id(0);                             \n"  \
    "   const int group_size = get_local_size(0);                    \n"  \
    "   partial[lid] = dot(x[gid * 4].vec, y[gid * 4].vec);          \n"  \
    "   barrier(CLK_LOCAL_MEM_FENCE);                                \n"  \
    "   for (int i = group_size / 2; i > 0; i >>= 1) {               \n"  \
    "      if (lid < i) {                                            \n"  \
    "         partial[lid] += partial[lid + i];                      \n"  \
    "      }                                                         \n"  \
    "      barrier(CLK_LOCAL_MEM_FENCE);                             \n"  \
    "   }                                                            \n"  \
    "   const int group_id = get_group_id(0);                        \n"  \
    "   if (lid == 0) {                                              \n"  \
    "      z[group_id] = partial[0];                                 \n"  \
    "   }                                                            \n"  \
    "}\n"

#define kernel_name "dot_product_"

static const char* kernel_fp16 = kernel(kernel_name, "half");
static const char* kernel_fp32 = kernel(kernel_name, "float");
static const char* kernel_fp64 = kernel(kernel_name, "double");

static uint32_t seed = 1;

static cl_int x_dot_y(ocl_context_t* c, ocl_command_queue_t q, ocl_kernel_t k,
                      ocl_memory_t mx, ocl_memory_t my, ocl_memory_t mz,
                      int64_t items) {
    ocl_arg_t args[] =
        {{&mx, sizeof(ocl_memory_t)},
         {&my, sizeof(ocl_memory_t)},
         {&mz, sizeof(ocl_memory_t)},
         {null, items * 4 * sizeof(fp32_t)}
    };
    ocl_event_t completion = ocl.enqueue_range_kernel(c, q, k,
        N / 4, items, countof(args), args);
    ocl.wait(&completion, 1);
    ocl_profiling_t p = {0};
    ocl.profile(completion, &p, N);
    traceln("kernel execution on device: %.3f us Gops=%.3f",
            p.time * (1000 * 1000), p.gops); // microseconds
    ocl.dispose_event(completion);
    return 0;
}

static cl_int test(ocl_context_t* c) {
    cl_int result = 0;
    ocl_command_queue_t q = ocl.create_command_queue(c, true);
    const char* code = kernel_fp32;
    ocl_program_t p = ocl.compile_program(c, code, strlen(code));
    ocl_kernel_t k = ocl.create_kernel(p, kernel_name "float");
    int64_t max_groups = ocl.devices[c->device_index].max_groups;
    int64_t max_items = ocl.devices[c->device_index].max_items[0];
    const int64_t n = N / 4;
    assertion(N % 4 == 0 && 1 <= n && n <= max_groups);
    int64_t groups = min(n, max_groups);
    int64_t items = min(n / groups, 1);
    assertion(1 <= items && items <= max_items);
    ocl_memory_t mx = ocl.allocate_write(c, N * sizeof(fp32_t));
    ocl_memory_t my = ocl.allocate_write(c, N * sizeof(fp32_t));
    ocl_memory_t mz = ocl.allocate_read(c,  groups * sizeof(fp32_t));
    // No need to allocate partial because it local memory for group
    fp32_t sum = 0;
    {   // initialize pinned memory:
        fp32_t* x = ocl.map_write(q, mx, 0, N * sizeof(fp32_t));
        fp32_t* y = ocl.map_write(q, my, 0, N * sizeof(fp32_t));
        // two input vectors
        for (int32_t i = 0; i < N; i++) {
//          x[i] = (fp32_t)(i + 1);
//          y[i] = (fp32_t)(N - i + 1);
//          x[i] = (fp32_t)(random32(&seed) / (double)INT32_MAX);
//          y[i] = (fp32_t)(random32(&seed) / (double)INT32_MAX);
            x[i] = (fp32_t)(i + 1);
            y[i] = (fp32_t)(i + 1);
            sum = sum + x[i] * y[i];
        }
        ocl.unmap(q, mx, x);
        ocl.unmap(q, my, y);
    }
    x_dot_y(c, q, k, mx, my, mz, items);
    {   // map result and verify it:
        fp32_t* z = (fp32_t*)ocl.map_read(q, mz, 0, groups * sizeof(fp32_t));
        fp32_t dot = 0;
        for (int i = 0; i < groups; i++) { dot += z[i]; }
        traceln("dot: %.1f  sum: %.1f delta: %.7f\n", dot, sum, fabs(dot - sum));
        assertion(dot == sum, "dot_product():%.1f != %.1f\n", dot, sum);
        ocl.unmap(q, mz, z);
    }
    ocl.flush_command_queue(q);   // must be called before deallocate()
    ocl.finish_command_queue(q);  // must be called before deallocate()
    ocl.deallocate(mx); // must be dealloca() before dispose_command_queue()
    ocl.deallocate(my);
    ocl.deallocate(mz);
    ocl.dispose_kernel(k);
    ocl.dispose_program(p);
    ocl.dispose_command_queue(q);
    return result;
}

int32_t _main_(int32_t argc, const char* argv[]) {
    (void)argc; (void)argv;
    cl_int result = 0;
    ocl.init();
    for (int cycles = 1; cycles > 0; cycles--) {
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

/*
   Intel: Broadwell, Skylake, Kaby Lake, Coffee Lake, Ice Lake, Tiger Lake
   https://en.wikipedia.org/wiki/List_of_Intel_CPU_microarchitectures
   support float16 kernels
*/