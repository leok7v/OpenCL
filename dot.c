#include "rt.h"
#include <CL/opencl.h>
#include <assert.h>
#include <windows.h>
#include "ocl.h"

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

enum { N = 4096 };

#define fp32_t float
#define fp64_t double

#define cl_fp32_t "float2"
#define cl_fp64_t "float4"

#ifdef USE_FP64
#define fp_t    fp64_t
#define cl_fp_t cl_fp46_t
#else
#define fp_t    fp32_t
#define cl_fp_t cl_fp32_t
#endif

static cl_int x_dot_y(ocl_context_t* c, ocl_command_queue_t q, ocl_kernel_t k,
                      ocl_memory_t mx, ocl_memory_t my, ocl_memory_t mz,
                      int64_t groups, int64_t items_in_work_group) {
    fp_t sum = 0;
    {   // initialize pinned memory:
        fp_t* x = ocl.map_write(q, mx, 0, N * sizeof(fp_t));
        fp_t* y = ocl.map_write(q, my, 0, N * sizeof(fp_t));
        // two input vectors
        for (int32_t i = 0; i < N; i++) {
            x[i] = (fp_t)(i + 1);
            y[i] = (fp_t)(N - i + 1);
            sum = sum + x[i] * y[i];
        }
        ocl.unmap(q, mx, x);
        ocl.unmap(q, my, y);
    }
    ocl_arg_t args[] =
        {{&mx, sizeof(ocl_memory_t)},
         {&my, sizeof(ocl_memory_t)},
         {&mz, sizeof(ocl_memory_t)},
         {null, groups * 4 * sizeof(fp_t)}
    };
    ocl_event_t completion = ocl.enqueue_range_kernel(c, q, k, N, items_in_work_group,
        countof(args), args);
    ocl.wait(&completion, 1);
    ocl_profiling_t p = {0};
    ocl.profile(completion, &p, N);
    ocl.dispose_event(completion);
    {   // map result and verify it:
        fp_t* z = (fp_t*)ocl.map_read(q, mz, 0, groups * sizeof(fp_t));
        assertion(*z == sum, "dot_product():%.1f != %.1f\n", *z, sum);
        ocl.unmap(q, mz, z);
    }
    traceln("kernel execution on device: %.3f us (microseconds) Gops=%.3f",
            p.time * (1000 * 1000), p.gops);
    return 0;
}

#define kernel_name "dot_product"

static cl_int test(ocl_context_t* c) {
    cl_int result = 0;
    ocl_command_queue_t q = ocl.create_command_queue(c, true);
    static const char* code =
    "__kernel void " kernel_name "(__global " cl_fp_t "* x,          \n"
    "                              __global " cl_fp_t "* y,          \n"
    "                              __global " cl_fp_t "* z,          \n"
    "                              __local  " cl_fp_t "* partial) {  \n"
    "                                                                \n"
    "   int gid = get_global_id(0);                                  \n"
    "   int lid = get_local_id(0);                                   \n"
    "   int group_size = get_local_size(0);                          \n"
    "                                                                \n"
    "   partial[lid] = x[gid] * y[gid];                              \n"
    "   barrier(CLK_LOCAL_MEM_FENCE);                                \n"
    "                                                                \n"
    "   for (int i = group_size / 2; i > 0; i >>= 1) {               \n"
    "      if (lid < i) {                                            \n"
    "         partial[lid] += partial[lid + i];                      \n"
    "      }                                                         \n"
    "      barrier(CLK_LOCAL_MEM_FENCE);                             \n"
    "   }                                                            \n"
    "                                                                \n"
    "   if (lid == 0) {                                              \n"
    "      z[get_group_id(0)] = dot(partial[0], (" cl_fp_t ")(1.0)); \n"
    "   }                                                            \n"
    "}\n";
    ocl_program_t p = ocl.compile_program(c, code, strlen(code));
    ocl_kernel_t k = ocl.create_kernel(p, kernel_name);

    int64_t items_in_work_group =
        min(ocl.devices[c->device_index].max_work_group, N * 8);
    int64_t groups = N / (items_in_work_group * 4);
    assert(groups >= 1);
    ocl_memory_t mx = ocl.allocate_write(c, N * sizeof(fp_t));
    ocl_memory_t my = ocl.allocate_write(c, N * sizeof(fp_t));
    ocl_memory_t mz = ocl.allocate_read(c,  groups * sizeof(fp_t));
    x_dot_y(c, q, k, mx, my, mz, groups, items_in_work_group);
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
