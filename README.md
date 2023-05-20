# ocl - make OpenCL simple

- [x] Updated project to Visual Studio 2022
- [x] Do not need to instal Nvidia SDK or Intel SDK
- [x] OpenCL header used from: 
   https://github.com/KhronosGroup/OpenCL-Headers/tree/main/CL
- [x] OpenCL.dll exists on Windows and routes to both Intel and Nvidia drivers. But, possibly very old 1.2 version.
- [x] Generated binding using GetProcAddress and trivial heared files parsing.
- [x] Implemented 1-dimensional single command queue fail fast ocl.* interface.
- [x] Implemented trivial host fp16_t support
- [ ] Design gpu.* interface to unify OpenCL and possbily Cuda and/or DirectCompute?
- [ ] implement sum(v) measure perfromance of submitting to queue and reading results on host side
- [ ] test dot.c for a) 1..16 x 1..16 dot() b) huge dataset clustered around 1.0+/-delta c) measure all performances on add.c 
- [ ] implement gemv()
- [x] dot.c -> blast.c (Basic Linear Algebra Subrotines/Subprograms/Functions TINY)


