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
}
