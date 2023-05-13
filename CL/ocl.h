#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ocl_device_id_s*      ocl_device_id_t;
typedef struct ocl_command_queue_s*  ocl_command_queue_t;
typedef struct ocl_memory_s*         ocl_memory_t;
typedef struct ocl_program_s*        ocl_program_t;
typedef struct ocl_kernel_s*         ocl_kernel_t;
typedef struct ocl_event_s*          ocl_event_t;

typedef struct ocl_device_s {
    void* platform;
    ocl_device_id_t id; // device id
    char  name[128];
    char  vendor[128];
    int32_t version_major;
    int32_t version_minor;
    int64_t clock_frequency; // MHz
    int64_t global_memory;   // size in bytes
    int64_t local_memory;    // size in bytes
    int64_t compute_units;   // max compute units
    int64_t max_work_group;       // max work group size, see: ** below
    int64_t work_item_dimensions; // dimensionality of work items
    int64_t work_items[3];        // work_item_sizes per dimension (0 means any?)
} ocl_device_t;

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
// usage of "size" and "max" is superconfusing in OpenCL docs this avoided here

typedef struct ocl_context_s {
    int32_t device_index;
    void*   ctx; // OpenCL context
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
    double  time; // seconds: end - start
    double  gops; // Giga items per second
} ocl_profiling_t;

typedef struct ocl_if {
    void  (*init)(void);
    void  (*open)(ocl_context_t* c, int32_t device_index);
    // profiling: true waits(!!) for kernel to finish and returns profiling info
    ocl_command_queue_t (*create_command_queue)(ocl_context_t* c, bool profiling);
    // pinned memory with CL_MEM_ALLOC_HOST_PTR
    ocl_memory_t (*allocate_read)(ocl_context_t* c, size_t bytes);
    ocl_memory_t (*allocate_write)(ocl_context_t* c, size_t bytes);
    ocl_memory_t (*allocate_rw)(ocl_context_t* c, size_t bytes);
    // both flush and finish must be called before deallocate()
    void  (*flush_command_queue)(ocl_command_queue_t command_queue);
    void  (*finish_command_queue)(ocl_command_queue_t command_queue);
    void (*deallocate)(ocl_memory_t m);
    // map_write - host will read data written by GPU
    // map_write - host will write data that GPU will read
    void* (*map_read)(ocl_command_queue_t q, ocl_memory_t m, size_t offset, size_t bytes);
    void* (*map_write)(ocl_command_queue_t q, ocl_memory_t m, size_t offset, size_t bytes);
    void* (*map_rw)(ocl_command_queue_t q, ocl_memory_t m, size_t offset, size_t bytes);
    // memory must be unmapped before the kernel is executed
    void  (*unmap)(ocl_command_queue_t q, ocl_memory_t m, void* address);
    ocl_program_t (*compile_program)(ocl_context_t* c, const char* code, size_t bytes);
    ocl_kernel_t (*create_kernel)(ocl_program_t p, const char* name);
    void (*kernel_info)(ocl_context_t* c, ocl_kernel_t kernel, ocl_kernel_info_t* info);
    // 1-dimensional range kernel: if items_in_work_group is 0 max is used
    ocl_event_t (*enqueue_range_kernel)(ocl_context_t* c, ocl_command_queue_t q,
        ocl_kernel_t k, size_t items, size_t items_in_work_group,
        int argc, ocl_arg_t argv[]);
    void (*wait)(ocl_event_t events[], int count);
    // must wait() first before calling profile()
    void (*profile)(ocl_event_t e, ocl_profiling_t* p, int64_t items);
    void (*dispose_event)(ocl_event_t e);
    const char* (*error)(int result);
    void  (*dispose_program)(ocl_program_t p);
    void  (*dispose_kernel)(ocl_kernel_t k);
    // dispose_command_queue() call after all deallocte()
    void  (*dispose_command_queue)(ocl_command_queue_t command_queue);
    void (*close)(ocl_context_t* c);
    ocl_device_t* devices;
    int32_t device_count;
} ocl_if;

extern ocl_if ocl;

#ifdef __cplusplus
}
#endif
