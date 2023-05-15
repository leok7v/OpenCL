// expectes to be prepended with
// #define name actual_kernel_name
// #define fp_t float
// or
// #define fp_t double
// #define fp_t half

// #pragma OPENCL SELECT_ROUNDING_MODE rte // rte rtz rtp rtn

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_fp16: enable

__kernel void name(__global const float* x,
                   __global const float* y,
                   __global float* r,
                   __local  float* partial,
                   const int offset,
                   const int stride0,
                   const int stride1) {
    const int gid = get_global_id(0);   // global id
    const int lid = get_local_id(0);    // local id
    const int group_size = get_local_size(0);  // local work group size
    partial[lid] = x[gid * stride0 + offset] * y[gid * stride1 + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
    int m = group_size;
    int n = m >> 1;
    while (n > 0) {
        if (lid == 0 && (m & 1)) { partial[lid] += partial[n << 1]; }
        if (lid < n) { partial[lid] += partial[lid + n]; }
        barrier(CLK_LOCAL_MEM_FENCE);
        m = n;
        n >>= 1;
    }
    if (lid == 0) {
        r[get_group_id(0)] = partial[0];
    }
}


// __kernel void name(__global const float* x,
//                    __global const float* y,
//                    __global float* r,
//                    __local  float* partial,
//                    const int offset,
//                    const int stride0,
//                    const int stride1) {
//     const int gid = get_global_id(0);   // global id
//     const int lid = get_local_id(0);    // local id
//     const int group_size = get_local_size(0);  // local work group size
//     partial[lid] = x[gid * stride0 + offset] * y[gid * stride1 + offset];
//     barrier(CLK_LOCAL_MEM_FENCE);
//     for (int i = group_size / 2; i > 0; i >>= 1) {
//         if (lid < i) { partial[lid] += partial[lid + i]; }
//         barrier(CLK_LOCAL_MEM_FENCE); // not inside "if"!
//     }
//     if (lid == 0) {
//         r[get_group_id(0)] = partial[0];
//     }
// }


//  if (lid == 0) {
//      float sum = 0.0f;
//      for (int i = 0; i < group_size; i++) { sum += partial[i]; }
//      r[get_group_id(0)] = partial[lid];
//  }

//  const int items_per_group = get_global_size(0) / group_size;
//  const int group_id = gid / items_per_group;
//  const int local_index = gid % items_per_group;
