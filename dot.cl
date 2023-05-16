// expectes to be prepended with
// #define suffix fp16|fp32|fp64
// #define fp_t float
// or
// #define fp_t double
// #define fp_t half

// #pragma OPENCL SELECT_ROUNDING_MODE rte // rte rtz rtp rtn

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_fp16: enable

#define _n_(f, l) f ##_## l
#define name(first, last) _n_(first, last)

__kernel void name(sum_odd, suffix)(
        __global const float* v, const int offset, const int stride,
        __global float* r) {
    const int i = get_global_id(0);
    const int m = get_global_size(0); // middle
    const int e = get_global_size(0) * 2; // end
    if (i == 0) {
        r[i] = v[offset + i * stride] + v[offset + (i + m) * stride] +
               v[offset + e * stride]; // extra one for odd
    } else {
        r[i] = v[offset + i * stride] + v[offset + (i + m) * stride];
    }
}

__kernel void name(sum_even, suffix)(
        __global const float* v, const int offset, const int stride,
        __global float* r) {
    const int i = get_global_id(0);
    const int m = get_global_size(0);
    r[i] = v[offset + i * stride] + v[offset + (i + m) * stride];
}

__kernel void name(dot, suffix)(
        __global const float* v0, const int offset0, const int stride0,
        __global const float* v1, const int offset1, const int stride1,
        __global float* r) {
    const int i = get_global_id(0); // (0) of dimension zero out of 3
    r[i] = v0[offset0 + i * stride0] * v1[offset1 + i * stride1];
}
