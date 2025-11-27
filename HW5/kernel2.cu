#include <cstdio>
#include <cstdlib>
#include <cuda.h>

// Each CUDA thread processes one pixel. Use cudaHostAlloc to allocate the host memory, and use
// cudaMallocPitch to allocate GPU memory. Name the file kernel2.cu.
__global__ void mandel_kernel()
{
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int *img,
             int res_x,
             int res_y,
             int max_iterations)
{
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;
}
