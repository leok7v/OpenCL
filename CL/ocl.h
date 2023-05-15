#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ocl_device_id_s* ocl_device_id_t;
typedef struct ocl_memory_s*    ocl_memory_t;
typedef struct ocl_program_s*   ocl_program_t;
typedef struct ocl_kernel_s*    ocl_kernel_t;
typedef struct ocl_event_s*     ocl_event_t;

enum { // float_fp_config, doublefp_config bits
    ocl_fp_denorm                        = (1 << 0),
    ocl_fp_inf_nan                       = (1 << 1),
    ocl_fp_round_to_nearest              = (1 << 2),
    ocl_fp_round_to_zero                 = (1 << 3),
    ocl_fp_round_to_inf                  = (1 << 4),
    ocl_fp_fma                           = (1 << 5),
    ocl_fp_soft_float                    = (1 << 6),
    ocl_fp_correctly_rounded_divide_sqrt = (1 << 7)
};
// __kernel can use
// #pragma OPENCL SELECT_ROUNDING_MODE rte // rte rtz rtp rtn
// and
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable
// #pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef struct ocl_device_s {
    void* platform;
    ocl_device_id_t id; // device id
    char  name[128];
    char  vendor[128];
    int32_t version_major;    // OpenCL version
    int32_t version_minor;
    int32_t c_version_major;  // OpenCL kernel .cl C language version
    int32_t c_version_minor;  // see: note ** below
    int64_t clock_frequency;  // MHz
    int64_t global_memory;    // size in bytes
    int64_t local_memory;     // size in bytes
    int64_t compute_units;    // max compute units, see: *** below
    int64_t max_groups;       // max number of work groups, see: ** below
    int64_t dimensions;       // dimensionality of work items
    int64_t max_items[3];     // max work items in a group per dimension
    int64_t double_fp_config;
    int64_t float_fp_config;
} ocl_device_t;

// ** confusion between CL and CL_C versions:
// https://stackoverflow.com/questions/67371938/nvidia-opencl-device-version

// *** NVIDIA GeForce RTX 3080 Laptop reports 48 units, but
// These units are often referred to as "compute cores", or
// "streaming multiprocessors" (SMs) or "CUDA cores"
// For the GeForce RTX 3080, the CUDA core count is 8704.

typedef struct ocl_kernel_info_s { // CL_KERNEL_*
    int64_t work_group;      // max kernel work group size
    int64_t compile_work_group;
    int64_t local_memory;
    int64_t preferred_work_group_multiple;
    int64_t private_mem_size;
    int64_t global_work_size;
} ocl_kernel_info_t;

// ** work groups and items:
// https://stackoverflow.com/questions/62236072/understanding-cl-device-max-work-group-size-limit-opencl
// https://registry.khronos.org/OpenCL/sdk/2.2/docs/man/html/clEnqueueNDRangeKernel.html
// usage of "size" "max" is confusing in OpenCL docs this avoided here

typedef struct ocl_context_s {
    int32_t ix; // device index
    void*   c; // OpenCL context
    void*   q; // OpenCL command queue
    bool    profile; // cannot change on the fly
} ocl_context_t;

typedef struct ocl_arg_s {
    void* p;
    size_t bytes;
} ocl_arg_t;

typedef struct ocl_profiling_s { // in nanoseconds
    uint64_t queued;
    uint64_t submit;
    uint64_t start;
    uint64_t end;
    double  user; // seconds: host time (to be filled by client)
    double  time; // seconds: end - start
    double  gops; // Giga items per second
    uint64_t ema_samples; // 0 defaults to 128 samples
    struct { // exponential moving average
        double  user; // seconds: host time (to be calculated by client)
        double  time; // seconds: end - start
        double  gops; // Giga items per second
    } ema;
} ocl_profiling_t;

enum { // .allocate() access flags (matching OpenCL)
    ocl_allocate_read  = (1 << 2),
    ocl_allocate_write = (1 << 1),
    ocl_allocate_rw    = (1 << 0)
};

enum { // .map() access flags (matching OpenCL)
    ocl_map_read  = (1 << 0),
    ocl_map_write = (1 << 2), // invalidates region
    ocl_map_rw    = ((1 << 0) | (1 << 1)),
};

// single device single queue OpenCL interface

typedef struct ocl_if {
    void (*init)(void); // initializes devices[count] array
    void (*dump)(int ix); // dumps device info
    void (*open)(ocl_context_t* c, int32_t ix, bool profiling);
    // pinned memory with CL_MEM_ALLOC_HOST_PTR
    ocl_memory_t (*allocate)(ocl_context_t* c, int access, size_t bytes);
    void (*flush)(ocl_context_t* c); // all queued command to GPU
    void (*finish)(ocl_context_t* c); // waits for all commands to finish
    void (*deallocate)(ocl_memory_t m);
    // ocl_map_read  - host will read data written by GPU
    // ocl_map_write - host will write data that GPU will read
    void* (*map)(ocl_context_t* c, int flags, ocl_memory_t m,
        size_t offset, size_t bytes);
    // memory must be unmapped before the kernel is executed
    void (*unmap)(ocl_context_t* c, ocl_memory_t m, const void* address);
    ocl_program_t (*compile_program)(ocl_context_t* c, const char* code,
        size_t bytes);
    ocl_kernel_t (*create_kernel)(ocl_program_t p, const char* name);
    void (*kernel_info)(ocl_context_t* c, ocl_kernel_t kernel,
        ocl_kernel_info_t* info);
    // 1-dimensional range kernel: if items_in_work_group is 0 max is used
    ocl_event_t (*enqueue_range_kernel)(ocl_context_t* c, ocl_kernel_t k,
        size_t groups, size_t items,
        int argc, ocl_arg_t argv[]);
    void (*wait)(ocl_event_t* events, int count);
    // must wait() first before calling profile()
    void (*profile)(ocl_event_t e, ocl_profiling_t* p, int64_t items);
    void (*dispose_event)(ocl_event_t e);
    const char* (*error)(int result);
    void  (*dispose_program)(ocl_program_t p);
    void  (*dispose_kernel)(ocl_kernel_t k);
    void (*close)(ocl_context_t* c);
    ocl_device_t* devices;
    int32_t count;
} ocl_if;

extern ocl_if ocl;

#ifdef __cplusplus
}
#endif

/*
    In OpenCL, a work-item is a single unit of work that can be executed in
  parallel by a processing element. Each work-item is assigned a unique
  identifier within its work-group. Work-groups are collections of
  work-items that are executed together on a single processing element,
  such as a GPU core.

    The total number of work-items is determined by the global work-size,
  which is specified when the kernel is launched using clEnqueueNDRangeKernel.
  The global work-size is divided into work-groups of a fixed size specified
  by the local work-size. The number of work-groups is equal to the global
  work-size divided by the local work-size.

    For example, if the global work-size is (1024, 1024) and the local
  work-size is (8, 8), there will be 128 x 128 work-groups, each
  consisting of 8 x 8 = 64 work-items.

    Within a work-group, work-items can communicate with each other using
  local memory. Local memory is a shared memory space that is accessible
  only to work-items within the same work-group. Work-items within a
  work-group can synchronize their execution using barriers, which ensure
  that all work-items have completed their previous work-items before
  continuing execution.

  enqueue_range_kernel is 1-dimensional version of clEnqueueNDRangeKernel.
  enqueue_range_kernel_2D/3D can be exposed if needed.
*/