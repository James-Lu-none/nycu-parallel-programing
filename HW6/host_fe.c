#include "host_fe.h"
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

// host_fe() is the host front-end function that allocates memories and launches a GPU kernel,
// called convolution(), which is located in kernel.cl.
//Currently host_fe() and convolution() do not do any computation
// and return immediately.You should complete these two functions to accomplish
//    this assignment.

void host_fe(int filter_width,
             float *filter,
             int image_height,
             int image_width,
             float *input_image,
             float *output_image,
             cl_device_id *device,
             cl_context *context,
             cl_program *program)
{
    cl_int status;
    int filter_size = filter_width * filter_width;
    // Sizes in bytes
    const size_t image_bytes = (size_t)image_height * (size_t)image_width * sizeof(float);
    const size_t filter_bytes = (size_t)filter_size * sizeof(float);

    // Create buffers
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, filter_bytes,
                                     filter, &status);
    CHECK(status, "clCreateBuffer d_filter");

    cl_mem d_input = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_bytes,
                                    input_image, &status);
    CHECK(status, "clCreateBuffer d_input");

    cl_mem d_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_bytes, NULL, &status);
    CHECK(status, "clCreateBuffer d_output");

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);
    CHECK(status, "clCreateCommandQueue");

    // Create kernel from program
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    CHECK(status, "clCreateKernel");

    // Set kernel arguments to match kernel.cl signature
    status = clSetKernelArg(kernel, 0, sizeof(int), &filter_width);
    CHECK(status, "clSetKernelArg 0");
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
    CHECK(status, "clSetKernelArg 1");
    status = clSetKernelArg(kernel, 2, sizeof(int), &image_height);
    CHECK(status, "clSetKernelArg 2");
    status = clSetKernelArg(kernel, 3, sizeof(int), &image_width);
    CHECK(status, "clSetKernelArg 3");
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_input);
    CHECK(status, "clSetKernelArg 4");
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_output);
    CHECK(status, "clSetKernelArg 5");

    // Launch kernel with 2D NDRange: [image_height, image_width]
    size_t global_work_size[2];
    global_work_size[0] = (size_t)image_height;
    global_work_size[1] = (size_t)image_width;

    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    // Read back results
    status = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, image_bytes, output_image, 0, NULL, NULL);
    CHECK(status, "clEnqueueReadBuffer");

    // Finish and release resources
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
}
