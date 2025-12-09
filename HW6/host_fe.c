#include "host_fe.h"
#include "helper.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    static float *cached_filter = NULL;
    static int cached_filter_width = 0;
    static int cached_new_width = 0;
    static cl_mem cached_d_filter = NULL;

    if (cached_filter == NULL || filter != cached_filter || filter_width != cached_filter_width)
    {
        if (cached_d_filter)
            clReleaseMemObject(cached_d_filter);
        if (cached_filter)
            free(cached_filter);

        int max_size = filter_width * filter_width;
        float *work_filter = (float *)malloc(max_size * sizeof(float));
        memcpy(work_filter, filter, max_size * sizeof(float));

        int new_filter_width = filter_width;

        while (new_filter_width > 1)
        {
            int all_zeros = 1;
            int w = new_filter_width;
            int last = w - 1;

            for (int i = 0; i < w; ++i)
            {
                if (work_filter[i] != 0.0f || work_filter[last * w + i] != 0.0f
                    || work_filter[i * w] != 0.0f || work_filter[i * w + last] != 0.0f)
                {
                    all_zeros = 0;
                    break;
                }
            }

            if (!all_zeros)
                break;

            int old_width = new_filter_width;
            new_filter_width -= 2;

            for (int i = 0; i < new_filter_width; ++i)
                memcpy(&work_filter[i * new_filter_width], &work_filter[(i + 1) * old_width + 1],
                       new_filter_width * sizeof(float));
        }

        size_t filter_bytes = (size_t)(new_filter_width * new_filter_width) * sizeof(float);
        cl_int status;
        cached_d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         filter_bytes, work_filter, &status);

        cached_filter = filter;
        cached_filter_width = filter_width;
        cached_new_width = new_filter_width;
        free(work_filter);
    }

    int new_filter_width = cached_new_width;

    cl_int status;
    size_t image_bytes = (size_t)image_height * (size_t)image_width * sizeof(float);

    cl_mem d_input = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_bytes,
                                    input_image, &status);
    cl_mem d_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_bytes, NULL, &status);

    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);

    const char *kernel_name = (new_filter_width == 3)   ? "convolution_3x3"
                                                        : "convolution";

    cl_kernel kernel = clCreateKernel(*program, kernel_name, &status);

    clSetKernelArg(kernel, 0, sizeof(int), &new_filter_width);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &cached_d_filter);
    clSetKernelArg(kernel, 2, sizeof(int), &image_height);
    clSetKernelArg(kernel, 3, sizeof(int), &image_width);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_output);

    const int BLOCK_SIZE = 16;
    int R = new_filter_width >> 1;
    size_t tile_size = (size_t)(BLOCK_SIZE + (R << 1));
    size_t local_tile_bytes = tile_size * tile_size * sizeof(float);

    clSetKernelArg(kernel, 6, local_tile_bytes, NULL);

    size_t local_work_size[2] = {tile_size, tile_size};
    size_t global_work_size[2] = {((image_height + BLOCK_SIZE - 1) / BLOCK_SIZE) * tile_size,
                                  ((image_width + BLOCK_SIZE - 1) / BLOCK_SIZE) * tile_size};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL,
                           NULL);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, image_bytes, output_image, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
}
