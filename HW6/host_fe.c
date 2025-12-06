#include "host_fe.h"
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

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
    // simplify the filter if outer values are zero
    int max_size = filter_width * filter_width;
    float *work_filter = (float *)malloc(max_size * sizeof(float));

    memcpy(work_filter, filter, max_size * sizeof(float));

    int new_filter_width = filter_width;

    while (new_filter_width > 1)
    {
        int all_zeros = 1;
        for (int i = 0; i < new_filter_width; ++i)
        {
            if (work_filter[i] != 0.0f // top row
                || work_filter[(new_filter_width - 1) * new_filter_width + i] != 0.0f // bottom row
                || work_filter[i * new_filter_width] != 0.0f // left column
                || work_filter[i * new_filter_width + new_filter_width - 1] != 0.0f) // right column
            {
                all_zeros = 0;
                break;
            }
        }
        if (!all_zeros)
            break;
        int old_width = new_filter_width;
        new_filter_width -= 2;
        
        // shift filter values inward
        for (int i = 0; i < new_filter_width; ++i)
        {
            for (int j = 0; j < new_filter_width; ++j)
            {
                work_filter[i * new_filter_width + j] = work_filter[(i + 1) * old_width + (j + 1)];
            }
        }
    }

    cl_int status;
    int filter_size = new_filter_width * new_filter_width;
    const size_t image_bytes = (size_t)image_height * (size_t)image_width * sizeof(float);
    const size_t filter_bytes = (size_t)filter_size * sizeof(float);

    // Create buffers

    // filter loads to constant memory __constant const float *filter
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, filter_bytes, work_filter, &status);
    CHECK(status, "clCreateBuffer d_filter");

    // input and output images load to global memory __global const float *input_image
    cl_mem d_input = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_bytes, input_image, &status);
    CHECK(status, "clCreateBuffer d_input");
    cl_mem d_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_bytes, NULL, &status);
    CHECK(status, "clCreateBuffer d_output");

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);
    CHECK(status, "clCreateCommandQueue");

    // Select the optimal kernel based on filter width
    const char *kernel_name;
    switch (new_filter_width)
    {
        case 3:
            kernel_name = "convolution_3x3";
            break;
        case 5:
            kernel_name = "convolution_5x5";
            break;
        case 7:
            kernel_name = "convolution_7x7";
            break;
        default:
            kernel_name = "convolution";
            break;
    }

    // Create kernel from program
    cl_kernel kernel = clCreateKernel(*program, kernel_name, &status);
    CHECK(status, "clCreateKernel");

    // Set kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(int), &new_filter_width);
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

    // Prepare local memory size for tile
    const int BLOCK_SIZE = 16;
    int R = new_filter_width / 2;
    size_t tile_size = (size_t)(BLOCK_SIZE + 2 * R);
    size_t local_tile_bytes = tile_size * tile_size * sizeof(float);

    // Set dynamic local memory argument
    status = clSetKernelArg(kernel, 6, local_tile_bytes, NULL);
    CHECK(status, "clSetKernelArg 6 (local)");

    // Launch kernel with 2D NDRange. Choose local work size BLOCK_SIZE x BLOCK_SIZE.
    size_t local_work_size[2] = { (size_t)BLOCK_SIZE, (size_t)BLOCK_SIZE };

    // Pad global sizes to multiples of local size
    size_t global_rows = ((size_t)image_height + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0];
    size_t global_cols = ((size_t)image_width + local_work_size[1] - 1) / local_work_size[1] * local_work_size[1];
    size_t global_work_size[2] = { global_rows, global_cols };

    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0,
                                    NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    // Read back results
    status = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, image_bytes, output_image, 0, NULL,
                                 NULL);
    CHECK(status, "clEnqueueReadBuffer");

    // Finish and release resources
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    free(work_filter);
}