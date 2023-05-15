#define PROGRAM_FILE "../dot_product.cl"
#define DOT_FUNC "dot_product"

// 2^18
#define VEC_SIZE 256*256

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/opencl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   }

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);
   }

   return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(ctx, 1,
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }
   return program;
}

int32_t _main_(int32_t argc, const char* argv[]);

int main(int32_t argc, const char* argv[]) {
    if (1) { return _main_(argc, argv); }
   /* Host/device data structures */
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel dot_kernel;
   size_t max_local_size, global_size;
   cl_int i, err;
   cl_uint num_groups;
   cl_event prof_event;
   cl_ulong time_start, time_end;

   /* Data and buffers */
   static float a_vec[VEC_SIZE];
   static float b_vec[VEC_SIZE];
   float dot_output, dot_check, result;
   float *output_vec;
   cl_mem a_buffer, b_buffer, output_buffer;

   /* Initialize input vectors */
   srand((unsigned int)time(0));
   for(i=0; i<VEC_SIZE; i++) {
      a_vec[i] = (float)rand()/RAND_MAX;
   }
   srand((unsigned int)time(0));
   for(i=0; i<VEC_SIZE; i++) {
      b_vec[i] = (float)rand()/RAND_MAX;
   }

   /* Initialize check value */
   dot_check = 0.0f;
   for(i=0; i<VEC_SIZE; i++) {
      dot_check += a_vec[i] * b_vec[i];
   }
   /* Create a device and context */
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);
   }

   /* Allocate output vector - one element for each work-group */
   clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
         sizeof(max_local_size), &max_local_size, NULL);
   num_groups = (cl_uint)((VEC_SIZE / 4) / max_local_size);
   output_vec = (float*) malloc(num_groups * sizeof(float));

   /* Build the program */
   program = build_program(context, device, PROGRAM_FILE);

   /* Create a kernel for the multiplication function */
   dot_kernel = clCreateKernel(program, DOT_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   /* Create buffers */
   a_buffer = clCreateBuffer(context,
         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
         sizeof(a_vec), a_vec, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);
   };
   b_buffer = clCreateBuffer(context,
         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
         sizeof(b_vec), b_vec, &err);
   output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
         num_groups * sizeof(float), NULL, &err);

   /* Create a command queue */
   queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);
   };

   /* Create arguments for multiplication kernel */
   err = clSetKernelArg(dot_kernel, 0, sizeof(cl_mem), &a_buffer);
   err |= clSetKernelArg(dot_kernel, 1, sizeof(cl_mem), &b_buffer);
   err |= clSetKernelArg(dot_kernel, 2, sizeof(cl_mem), &output_buffer);
   err |= clSetKernelArg(dot_kernel, 3, max_local_size * 4 * sizeof(float), NULL);
   if(err < 0) {
      printf("Couldn't set an argument for the dot product kernel");
      exit(1);
   };

   /* Enqueue multiplication kernel */
   global_size = VEC_SIZE/4;
   err = clEnqueueNDRangeKernel(queue, dot_kernel, 1, NULL, &global_size,
         &max_local_size, 0, NULL, &prof_event);
   if(err < 0) {
      perror("Couldn't enqueue the dot product kernel");
      exit(1);
   }

   /* Read output buffer */
   err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
      num_groups * sizeof(float), output_vec, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   /* Get profiling information */
   clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,
      sizeof(time_start), &time_start, NULL);
   clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
      sizeof(time_end), &time_end, NULL);
   printf("On the device, the dot product kernel completed in %llu ns.\n",
      (time_end - time_start));

   /* Obtain output value */
   dot_output = 0.0f;
   for(i = 0; i < (int)num_groups; i++)
      dot_output += output_vec[i];

   /* Check result */
   result = (float)fabs(dot_output - dot_check);
   if(result > 10.0f)
      printf("Dot product failed.\n");
   else
      printf("Dot product succeeded. Delta: %.17e\n", result);

   /* Deallocate resources */
   free(output_vec);
   clReleaseMemObject(a_buffer);
   clReleaseMemObject(b_buffer);
   clReleaseMemObject(output_buffer);
   clReleaseKernel(dot_kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
