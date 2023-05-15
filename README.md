# ocl - make OpenCL simple

- [x] Updated project to Visual Studio 2022
- [x] Do not need to instal Nvidia SDK or Intel SDK
- [x] OpenCL header used from: 
   https://github.com/KhronosGroup/OpenCL-Headers/tree/main/CL
- [x] OpenCL.dll exists on Windows and routes to both Intel and Nvidia drivers.
- [x] Generated binding using GetProcAddress and trivial heared files parsing.

## next step
https://github.com/mattscar/opencl_dot_product

## interesting kernels

https://github.com/pocl/pocl/blob/master/lib/kernel/dot.cl

### original code was at
https://github.com/pratikone/OpenCL-learning-VS2012

Configuration of Visual Studio 2012 was a struggle

https://medium.com/@pratikone/opencl-on-visual-studio-configuration-tutorial-for-the-confused-3ec1c2b5f0ca#.sr5v6xukd

"OpenCL on Visual Studio : Configuration tutorial for the confused"

none of that is needed at all.
