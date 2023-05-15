// expectes to be prepended with
// #define name actual_kernel_name
// #define fp_t float
// or
// #define fp_t double
// #define fp_t half

// #pragma OPENCL SELECT_ROUNDING_MODE rte // rte rtz rtp rtn

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_fp16: enable

__kernel void sum_odd_fp32(__global const float* x,
                           __global float* r) {
    const int gid = get_global_id(0);
    const int mid = get_global_size(0);
    const int end = get_global_size(0) * 2;
    if (gid == 0) {
        r[gid] = x[gid] + x[gid + mid] + x[end];
    } else {
        r[gid] = x[gid] + x[gid + mid];
    }
}

__kernel void sum_even_fp32(__global const float* x,
                            __global float* r) {
    const int gid = get_global_id(0);
    const int mid = get_global_size(0);
    r[gid] = x[gid] + x[gid + mid];
}

__kernel void name(__global const float* x,
                   __global const float* y,
                   __global float* r,
                   const int offset,
                   const int stride0,
                   const int stride1) {
    const int gid = get_global_id(0); // global id - dimension 0
    r[gid] = x[gid * stride0 + offset] * y[gid * stride1 + offset];
}
