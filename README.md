# ocl - make OpenCL simple

1. Updated project to Visual Studio 2022
2. Do not want to instal Nvidia SDK or Intel SDK
3. OpenCL header files are avaialbe at 
   https://github.com/KhronosGroup/OpenCL-Headers/tree/main/CL
4. OpenCL.dll exists on Windows and routes to both Intel and Nvidia drivers.
5. We need to generate binding using GetProcAddress and trivial heared files parsing.


# original code was at
https://github.com/pratikone/OpenCL-learning-VS2012

# Configuration in Visual Studio
https://medium.com/@pratikone/opencl-on-visual-studio-configuration-tutorial-for-the-confused-3ec1c2b5f0ca#.sr5v6xukd
