#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>


static inline int div_up(int a, int b)
{
    return (a + b - 1) / b;
}

__device__ __forceinline__ int mandel(float c_re, float c_im, int max_iterations)
{
    float z_re = c_re;
    float z_im = c_im;
    int i = 0;

    for (; i < max_iterations; ++i)
    {
        float zr2 = z_re * z_re;
        float zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.0f)
            break;

        float new_re = zr2 - zi2;
        float new_im = 2.0f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

// Template version: max_iterations known at compile time, enables aggressive unrolling
template <int MAX_IT> __device__ __forceinline__ int mandel_fixed(float c_re, float c_im)
{
    float z_re = c_re;
    float z_im = c_im;
    int i = 0;

#pragma unroll
    for (i = 0; i < MAX_IT; ++i)
    {
        float zr2 = z_re * z_re;
        float zi2 = z_im * z_im;
        if (zr2 + zi2 > 4.0f)
            break;

        float new_re = zr2 - zi2;
        float new_im = 2.0f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

// Dispatch to specialized version or fallback to generic
__device__ __forceinline__ int mandel_dispatch(float c_re, float c_im, int max_iterations)
{
    switch (max_iterations)
    {
        case 256:
            return mandel_fixed<256>(c_re, c_im);
        case 1000:
            return mandel_fixed<1000>(c_re, c_im);
        case 10000:
            return mandel_fixed<10000>(c_re, c_im);
        case 100000:
            return mandel_fixed<100000>(c_re, c_im);
        default:
            return mandel(c_re, c_im, max_iterations);
    }
}

__global__ void mandel_kernel(float lower_x,
                              float lower_y,
                              float step_x,
                              float step_y,
                              int max_iterations,
                              int res_x,
                              int res_y,
                              size_t pitch_bytes,
                              int *output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= res_x || y >= res_y)
        return;

    float c_re = lower_x + x * step_x;
    float c_im = lower_y + y * step_y;

    int it = mandel_dispatch(c_re, c_im, max_iterations);

    // Pitch is in bytes - get pointer to row y
    int *row = (int *)((char *)output + y * pitch_bytes);
    row[x] = it;
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

    int *device_img = nullptr;
    size_t pitch = 0;
    cudaMallocPitch((void **)&device_img, &pitch, res_x * sizeof(int), res_y);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    dim3 block(32, 32, 1);
    dim3 grid(div_up(res_x, block.x), div_up(res_y, block.y), 1);

    mandel_kernel<<<grid, block>>>(lower_x, lower_y, step_x, step_y, max_iterations, res_x, res_y, pitch, device_img);

    size_t img_size = (size_t)res_x * (size_t)res_y * sizeof(int);
    int *host_img = nullptr;
    cudaHostAlloc((void **)&host_img, img_size, cudaHostAllocDefault);

    cudaMemcpy2D(host_img, res_x * sizeof(int), device_img, pitch, res_x * sizeof(int), res_y, cudaMemcpyDeviceToHost);

    std::memcpy(img, host_img, img_size);

    cudaFree(device_img);
    cudaFreeHost(host_img);
}