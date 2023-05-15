#include "rt.h"
#include <CL/opencl.h>
#include <assert.h>
#include <windows.h>
#include "ocl.h"
#include <CL/cl_bind.inc> // dynamically bind everything

static ocl_device_t ocl_devices[32]; // up to 32 GPUs supported

enum { KB = 1024, MB = 1024 * KB, GB = 1024 * MB };

#define call(f) do { /* fail fast OpenCL API call */           \
    cl_int __err__;                                            \
    fatal_if((__err__  = (f)) != 0, "%s", ocl.error(__err__)); \
} while (0)

#define not_null(p, r) do {                             \
    fatal_if(r != 0 || p == null, "%s", ocl.error(r));  \
} while (0)

static void ocl_error_notify(const char * errinfo,
    const void* private_info, size_t cb, void* user_data) {
    traceln("ERROR: %*s", errinfo);
    (void)private_info;
    (void)cb;
    (void)user_data;
}

static void ocl_open(ocl_context_t* c, int32_t device_index) {
    call(!(0 <= device_index && device_index < ocl.device_count));
    ocl_device_t* d = &ocl.devices[device_index];
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)d->platform, 0
    };
    cl_int r = 0;
    cl_device_id id = (cl_device_id)d->id;
    cl_context ctx = clCreateContext(properties, 1, &id, ocl_error_notify,
        /* user_data: */ null, &r);
    not_null(ctx, r);
    c->ctx = ctx;
    c->device_index = device_index;
}

static ocl_queue_t ocl_create_queue(ocl_context_t* c,
        bool profiling) {
    cl_context ctx = c->ctx;
    cl_device_id device_id = (cl_device_id)ocl.devices[c->device_index].id;
    cl_int r = 0;
    static cl_command_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, device_id,
            profiling ? properties : null, &r);
    not_null(q, r);
    return (ocl_queue_t)q;
}

static void ocl_flush(ocl_queue_t q) {
    call(clFlush((cl_command_queue)q));
}

static void ocl_finish(ocl_queue_t q) {
    call(clFinish((cl_command_queue)q));
}

static void ocl_dispose_queue(ocl_queue_t q) {
    call(clReleaseCommandQueue((cl_command_queue)q));
}

// https://streamhpc.com/blog/2013-02-03/opencl-basics-flags-for-the-creating-memory-objects/
// https://man.opencl.org/clCreateBuffer.html

static ocl_memory_t ocl_allocate(ocl_context_t* c, int flags, size_t bytes) {
    cl_int r = 0;
    cl_mem m = clCreateBuffer(c->ctx, flags|CL_MEM_ALLOC_HOST_PTR, bytes, null, &r);
    not_null(m, r);
    return (ocl_memory_t)m;
}

static void  ocl_deallocate(ocl_memory_t m) {
    call(clReleaseMemObject((cl_mem)m));
}

static void* ocl_map(ocl_queue_t q, int flags, ocl_memory_t m, size_t offset,
        size_t bytes) {
    cl_int r = 0;
    // blocking_map: true sync mapping
    void* a = clEnqueueMapBuffer((cl_command_queue)q, (cl_mem)m,
        /*blocking_map: */ true, flags, offset, bytes, 0, null, null, &r);
    not_null(a, r);
    return a;
}

static void  ocl_unmap(ocl_queue_t q, ocl_memory_t m, const void* a) {
    call(clEnqueueUnmapMemObject((cl_command_queue)q, (cl_mem)m, (void*)a,
        0, null, null));
}

static ocl_program_t ocl_compile_program(ocl_context_t* c, const char* code, size_t bytes) {
    cl_int r = 0;
    cl_program p = clCreateProgramWithSource(c->ctx, 1, &code, &bytes, &r);
    not_null(p, r);
    // Build the program
    cl_device_id device_id = (cl_device_id)ocl.devices[c->device_index].id;
    r = clBuildProgram(p, 1, &device_id, null, null, null);
    if (r != 0) {
        traceln("clBuildProgram() failed %s", ocl.error(r));
        static char log[16 * 1024];
        log[0] = 0;
        call(clGetProgramBuildInfo(p, device_id, CL_PROGRAM_BUILD_LOG, countof(log), log, null) != 0);
        traceln("%s", log);
        fatal_if(true, "\n%s", ocl.error(r));
    }
    return (ocl_program_t)p;
}

static void ocl_dispose_program(ocl_program_t p) {
    call(clReleaseProgram((cl_program)p));
}

static ocl_kernel_t ocl_create_kernel(ocl_program_t p, const char* name) {
    cl_int r = 0;
    cl_kernel k = clCreateKernel((cl_program)p, name, &r);
    not_null(k, r);
    return (ocl_kernel_t)k;
}

static ocl_event_t ocl_enqueue_range_kernel(ocl_context_t* c, ocl_queue_t q,
        ocl_kernel_t k, size_t groups, size_t items_per_group,
        int argc, ocl_arg_t argv[]) {
    for (int i = 0; i < argc; i++) {
        call(clSetKernelArg((cl_kernel)k, i, argv[i].bytes, argv[i].p));
    }
    cl_event completion = null;
    size_t total = groups * items_per_group;
    ocl_device_t* d = &ocl.devices[c->device_index]; (void)d;
    assertion((int64_t)groups <= d->max_groups);
    assertion((int64_t)items_per_group <= d->max_items[0]);
    call(clEnqueueNDRangeKernel((cl_command_queue)q, (cl_kernel)k,
            1, null, &total, &items_per_group, 0, null, &completion));
    return (ocl_event_t)completion;
}

static void ocl_profile(ocl_event_t e, ocl_profiling_t* p, int64_t items) {
    #pragma push_macro("get_info")
    #define get_info(n, v) do { \
        call(clGetEventProfilingInfo((cl_event)e, n, sizeof(v), &v, null)); \
    } while (0)
    get_info(CL_PROFILING_COMMAND_QUEUED, p->queued);
    get_info(CL_PROFILING_COMMAND_SUBMIT, p->submit);
    get_info(CL_PROFILING_COMMAND_START, p->start);
    get_info(CL_PROFILING_COMMAND_END, p->end);
    #pragma pop_macro("get_info")
    p->time = (p->end - p->start) / (double)NSEC_IN_SEC;
    if (items != 0) {
        double seconds_per_item = p->time / items;
        double ops_per_second = 1.0 / seconds_per_item;
        p->gops = ops_per_second / (1000 * 1000 * 1000);
    } else {
        p->gops = 0; // cannot determine for unknown number of items
    }
}

static void ocl_wait(ocl_event_t* events, int count) {
    call(clWaitForEvents(count, (cl_event*)events));
}

static void ocl_dispose_event(ocl_event_t e) {
    call(clReleaseEvent((cl_event)e));
}

static void ocl_dispose_kernel(ocl_kernel_t k) {
    call(clReleaseKernel((cl_kernel)k));
}

static void ocl_kernel_info(ocl_context_t* c, ocl_kernel_t kernel,
        ocl_kernel_info_t* info) {
    cl_kernel k = (cl_kernel)kernel;
    cl_device_id device_id = (cl_device_id)ocl.devices[c->device_index].id;
    #pragma push_macro("get_val")
    #define get_val(n, v) do { \
        call(clGetKernelWorkGroupInfo(k, device_id, n, sizeof(v), &v, null)); \
    } while (0)
    get_val(CL_KERNEL_WORK_GROUP_SIZE, info->work_group);
    get_val(CL_KERNEL_LOCAL_MEM_SIZE, info->local_memory);
    get_val(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        info->preferred_work_group_multiple);
    get_val(CL_KERNEL_PRIVATE_MEM_SIZE, info->private_mem_size);
    get_val(CL_KERNEL_GLOBAL_WORK_SIZE, info->global_work_size);
    #pragma pop_macro("get_val")
}

static void ocl_close(ocl_context_t* c) {
    call(clReleaseContext((cl_context)c->ctx));
    c->ctx = null;
}

static const char* ocl_error(int r) {
    static char error[128];
    #define case_(x) case x: snprintf(error, countof(error), "%d " #x, r); break
    switch (r) {
        case_(CL_DEVICE_NOT_FOUND);
        case_(CL_DEVICE_NOT_AVAILABLE);
        case_(CL_COMPILER_NOT_AVAILABLE);
        case_(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        case_(CL_OUT_OF_RESOURCES);
        case_(CL_OUT_OF_HOST_MEMORY);
        case_(CL_PROFILING_INFO_NOT_AVAILABLE);
        case_(CL_MEM_COPY_OVERLAP);
        case_(CL_IMAGE_FORMAT_MISMATCH);
        case_(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        case_(CL_BUILD_PROGRAM_FAILURE);
        case_(CL_MAP_FAILURE);
        case_(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        case_(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        case_(CL_COMPILE_PROGRAM_FAILURE);
        case_(CL_LINKER_NOT_AVAILABLE);
        case_(CL_LINK_PROGRAM_FAILURE);
        case_(CL_DEVICE_PARTITION_FAILED);
        case_(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
        case_(CL_INVALID_VALUE);
        case_(CL_INVALID_DEVICE_TYPE);
        case_(CL_INVALID_PLATFORM);
        case_(CL_INVALID_DEVICE);
        case_(CL_INVALID_CONTEXT);
        case_(CL_INVALID_QUEUE_PROPERTIES);
        case_(CL_INVALID_COMMAND_QUEUE);
        case_(CL_INVALID_HOST_PTR);
        case_(CL_INVALID_MEM_OBJECT);
        case_(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        case_(CL_INVALID_IMAGE_SIZE);
        case_(CL_INVALID_SAMPLER);
        case_(CL_INVALID_BINARY);
        case_(CL_INVALID_BUILD_OPTIONS);
        case_(CL_INVALID_PROGRAM);
        case_(CL_INVALID_PROGRAM_EXECUTABLE);
        case_(CL_INVALID_KERNEL_NAME);
        case_(CL_INVALID_KERNEL_DEFINITION);
        case_(CL_INVALID_KERNEL);
        case_(CL_INVALID_ARG_INDEX);
        case_(CL_INVALID_ARG_VALUE);
        case_(CL_INVALID_ARG_SIZE);
        case_(CL_INVALID_KERNEL_ARGS);
        case_(CL_INVALID_WORK_DIMENSION);
        case_(CL_INVALID_WORK_GROUP_SIZE);
        case_(CL_INVALID_WORK_ITEM_SIZE);
        case_(CL_INVALID_GLOBAL_OFFSET);
        case_(CL_INVALID_EVENT_WAIT_LIST);
        case_(CL_INVALID_EVENT);
        case_(CL_INVALID_OPERATION);
        case_(CL_INVALID_GL_OBJECT);
        case_(CL_INVALID_BUFFER_SIZE);
        case_(CL_INVALID_MIP_LEVEL);
        case_(CL_INVALID_GLOBAL_WORK_SIZE);
        case_(CL_INVALID_PROPERTY);
        case_(CL_INVALID_IMAGE_DESCRIPTOR);
        case_(CL_INVALID_COMPILER_OPTIONS);
        case_(CL_INVALID_LINKER_OPTIONS);
        case_(CL_INVALID_DEVICE_PARTITION_COUNT);
        case_(CL_INVALID_PIPE_SIZE);
        case_(CL_INVALID_DEVICE_QUEUE);
        case_(CL_INVALID_SPEC_ID);
        case_(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
        default: snprintf(error, countof(error), "%d Unknown error", r);
    }
    error[countof(error) - 1] = 0;
    return error;
}

static void ocl_init(void) {
    #pragma push_macro("get_str")
    #pragma push_macro("get_val")
    #define get_str(name, s) do { \
        call(clGetDeviceInfo(id, name, countof(s), s, null)); \
    } while (0)
    #define get_val(name, v) do { \
        call(clGetDeviceInfo(id, name, sizeof(v), &v, null)); \
    } while (0)
    // Get platform and device information
    cl_platform_id platforms[16] = {0};
    cl_uint platform_count = countof(platforms);
	call(clGetPlatformIDs(platform_count, platforms, null));
    for (cl_uint i = 0; i < platform_count; i++) {
        cl_device_id device_ids[16] = {0};
        cl_uint devids_count = 0;
	    if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1,
                device_ids, &devids_count) == 0) {
            for (cl_uint j = 0; j < devids_count; j++) {
                ocl_device_t* d = &ocl.devices[ocl.device_count];
                cl_device_id id = device_ids[j];
                d->id = (ocl_device_id_t)id;
                d->platform = platforms[i];
                get_str(CL_DEVICE_NAME, d->name);
                get_str(CL_DEVICE_VENDOR, d->vendor);
                char version[128];
                get_str(CL_DEVICE_OPENCL_C_VERSION, version);
                int minor = 0; // sscanf wants type "int" not "int32_t"
                int major = 0;
                call(sscanf(version, "OpenCL C %d.%d", &major, &minor) != 2);
                d->version_major = major;
                d->version_minor = minor;
                get_str(CL_DEVICE_OPENCL_C_VERSION, version);
                call(sscanf(version, "OpenCL C %d.%d", &major, &minor) != 2);
                d->c_version_major = major;
                d->c_version_minor = minor;
                get_val(CL_DEVICE_MAX_CLOCK_FREQUENCY,      d->clock_frequency);
                get_val(CL_DEVICE_GLOBAL_MEM_SIZE,          d->global_memory);
                get_val(CL_DEVICE_LOCAL_MEM_SIZE,           d->local_memory);
                get_val(CL_DEVICE_MAX_COMPUTE_UNITS,        d->compute_units);
                get_val(CL_DEVICE_MAX_WORK_GROUP_SIZE,      d->max_groups);
                get_val(CL_DEVICE_DOUBLE_FP_CONFIG,         d->double_fp_config);
                get_val(CL_DEVICE_SINGLE_FP_CONFIG,         d->float_fp_config);
                get_val(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, d->dimensions);
                call(d->dimensions > countof(d->max_items));
                get_val(CL_DEVICE_MAX_WORK_ITEM_SIZES, d->max_items);
//              unfortunately this does NOT work:
//              #define CL_PLATFORM_ICD_SUFFIX_KHR 0x0920
//              char dll_pathname[260];
//              get_str(CL_PLATFORM_ICD_SUFFIX_KHR, dll_pathname);

                ocl.device_count++;
            }
        }
    }
    #pragma pop_macro("get_val")
    #pragma pop_macro("get_str")
}


static const char* ocl_fp_config_to_string(int64_t config) {
    static char s[1024];
    s[0] = 0;
    #pragma push_macro("append")
    #define append(text) do { strcat(s, ", " text); } while (0)
    if (config & ocl_fp_denorm)           { append("ocl_fp_denorm");           }
    if (config & ocl_fp_inf_nan)          { append("ocl_fp_inf_nan");          }
    if (config & ocl_fp_round_to_nearest) { append("ocl_fp_round_to_nearest"); }
    if (config & ocl_fp_round_to_zero)    { append("ocl_fp_round_to_zero");    }
    if (config & ocl_fp_round_to_inf)     { append("ocl_fp_round_to_inf");     }
    if (config & ocl_fp_fma)              { append("ocl_fp_fma");              }
    if (config & ocl_fp_soft_float)       { append("ocl_fp_soft_float");       }
    if (config & ocl_fp_correctly_rounded_divide_sqrt) {
        append("ocl_fp_correctly_rounded_divide_sqrt");
    }
    return s;
}

static void ocl_dump(int device_index) {
    const ocl_device_t* d = &ocl.devices[device_index];
    traceln("Device name:     %s OpenCL %d.%d C %d.%d", d->name,
        d->version_major, d->version_minor, d->c_version_major, d->c_version_minor);
    traceln("compute_units:    %lld @ %lldMHz", d->compute_units, d->clock_frequency);
    traceln("global_memory:    %lldMB", d->global_memory / MB);
    traceln("local_memory:     %lldMB", d->local_memory / MB);
    traceln("max_groups:       %lld", d->max_groups);
    traceln("dimensions:       %lld", d->dimensions);
    const int64_t* wi = d->max_items;
    traceln("max_items[]:     {%lld %lld %lld}", wi[0], wi[1], wi[2]);
    traceln("float_fp_config:  %s", ocl_fp_config_to_string(d->float_fp_config));
    traceln("double_fp_config: %s", ocl_fp_config_to_string(d->double_fp_config));
    #pragma pop_macro("append")
}

ocl_if ocl = {
    .init = ocl_init,
    .dump = ocl_dump,
    .open = ocl_open,
    .error = ocl_error,
    .create_queue = ocl_create_queue,
    .allocate = ocl_allocate,
    .deallocate = ocl_deallocate,
    .map = ocl_map,
    .unmap = ocl_unmap,
    .compile_program = ocl_compile_program,
    .create_kernel = ocl_create_kernel,
    .kernel_info = ocl_kernel_info,
    .enqueue_range_kernel = ocl_enqueue_range_kernel,
    .wait = ocl_wait,
    .profile = ocl_profile,
    .dispose_event = ocl_dispose_event,
    .dispose_kernel = ocl_dispose_kernel,
    .dispose_program = ocl_dispose_program,
    .flush = ocl_flush,
    .finish = ocl_finish,
    .dispose_queue = ocl_dispose_queue,
    .close = ocl_close,
    .devices = ocl_devices
};

// #pragma comment(lib, "OpenCL.lib")

static void* OpenCL;

void* clBindFunction(const char* name) {
    static bool init;
    if (!init) { OpenCL = LoadLibrary("OpenCL.dll"); init = true; }
    if (OpenCL != null) {
        return (void*)GetProcAddress(OpenCL, name);
    }
    return null;
}

static_assertion(CL_FP_DENORM                        == ocl_fp_denorm);
static_assertion(CL_FP_INF_NAN                       == ocl_fp_inf_nan);
static_assertion(CL_FP_ROUND_TO_NEAREST              == ocl_fp_round_to_nearest);
static_assertion(CL_FP_ROUND_TO_ZERO                 == ocl_fp_round_to_zero);
static_assertion(CL_FP_ROUND_TO_INF                  == ocl_fp_round_to_inf);
static_assertion(CL_FP_FMA                           == ocl_fp_fma);
static_assertion(CL_FP_SOFT_FLOAT                    == ocl_fp_soft_float);
static_assertion(CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT == ocl_fp_correctly_rounded_divide_sqrt);

static_assertion(CL_MEM_READ_WRITE == ocl_allocate_rw);
static_assertion(CL_MEM_WRITE_ONLY == ocl_allocate_write);
static_assertion(CL_MEM_READ_ONLY  == ocl_allocate_read);

static_assertion(CL_MAP_READ == ocl_map_read);
static_assertion((CL_MAP_WRITE|CL_MAP_READ) == ocl_map_rw);
static_assertion(CL_MAP_WRITE_INVALIDATE_REGION == ocl_map_write);

