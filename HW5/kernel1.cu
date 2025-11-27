#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>

__device__ int mandel(float c_re, float c_im, int max_iterations)
{
    float z_re = c_re;
    float z_im = c_im;
    int i = 0;
    for (; i < max_iterations; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;
        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandel_kernel(float lower_x,
                              float lower_y,
                              float step_x,
                              float step_y,
                              int max_iterations,
                              int res_x,
                              int res_y,
                              int *output)
{
    int this_x = blockIdx.x * blockDim.x + threadIdx.x;
    int this_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (this_x >= res_x || this_y >= res_y)
        return;

    float x = lower_x + this_x * step_x;
    float y = lower_y + this_y * step_y;
    int index = this_y * res_x + this_x;
    output[index] = mandel(x, y, max_iterations);
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

    int img_size = res_x * res_y * sizeof(int);
    int *host_img = new int[res_x * res_y];
    int *device_img = nullptr;
    
    // Allocate device memory
    cudaMalloc(&device_img, img_size);

    dim3 block(16, 16, 1);
    dim3 grid((res_x + block.x - 1) / block.x, (res_y + block.y - 1) / block.y);
    mandel_kernel<<<grid, block>>>(lower_x, lower_y, step_x, step_y, max_iterations, res_x, res_y, device_img);
    cudaMemcpy(host_img, device_img, img_size, cudaMemcpyDeviceToHost);

    std::memcpy(img, host_img, img_size);

    cudaFree(device_img);
    delete[] host_img;
}
