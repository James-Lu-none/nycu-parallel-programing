// Optimized tiled convolution kernel with loop unrolling
#define BLOCK_SIZE 16

// General purpose kernel with pragma unroll
__kernel void convolution(
    const int filter_width,
    __constant const float *filter,
    const int image_height,
    const int image_width,
    __global const float *input_image,
    __global float *output_image,
    __local float *local_tile)
{
    const int R = filter_width / 2;
    const int tile_size = BLOCK_SIZE + 2 * R;
    int lr = get_local_id(0);
    int lc = get_local_id(1);

    int gr = get_group_id(0);
    int gc = get_group_id(1);

    int group_row = gr * BLOCK_SIZE;
    int group_col = gc * BLOCK_SIZE;
    
    int tile_r = group_row + lr - R;
    int tile_c = group_col + lc - R;

    float v = 0.0f;
    if (tile_r >= 0 && tile_r < image_height &&
        tile_c >= 0 && tile_c < image_width)
    {
        v = input_image[tile_r * image_width + tile_c];
    }
    local_tile[lr * tile_size + lc] = v;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lr >= R && lr < R + BLOCK_SIZE &&
        lc >= R && lc < R + BLOCK_SIZE)
    {
        int out_r = group_row + (lr - R);
        int out_c = group_col + (lc - R);

        if (out_r < image_height && out_c < image_width)
        {
            float sum = 0.0f;

            #pragma unroll
            for (int fr = -R; fr <= R; fr++)
            {
                #pragma unroll
                for (int fc = -R; fc <= R; fc++)
                {
                    float image_value =
                        local_tile[(lr + fr) * tile_size + (lc + fc)];

                    float filter_value =
                        filter[(fr + R) * filter_width + (fc + R)];

                    sum += image_value * filter_value;
                }
            }

            output_image[out_r * image_width + out_c] = sum;
        }
    }
}

#define R 1  // Radius for 3x3 filter
#define TILE_SIZE (BLOCK_SIZE + 2 * R)

// Optimized tiled convolution kernel for 3x3 filter with loop unrolling
__kernel void convolution_3x3(
    const int filter_width,
    __constant const float *filter,
    const int image_height,
    const int image_width,
    __global const float *input_image,
    __global float *output_image,
    __local float *local_tile)
{
    int lr = get_local_id(0);
    int lc = get_local_id(1);
    int gr = get_group_id(0);
    int gc = get_group_id(1);
    int group_row = gr * BLOCK_SIZE;
    int group_col = gc * BLOCK_SIZE;
    int tile_r = group_row + lr - R;
    int tile_c = group_col + lc - R;
    float v = 0.0f;
    if (tile_r >= 0 && tile_r < image_height &&
        tile_c >= 0 && tile_c < image_width)
    {
        v = input_image[tile_r * image_width + tile_c];
    }
    local_tile[lr * TILE_SIZE + lc] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lr >= R && lr < R + BLOCK_SIZE && lc >= R && lc < R + BLOCK_SIZE)
    {
        int out_r = group_row + (lr - R);
        int out_c = group_col + (lc - R);
        if (out_r < image_height && out_c < image_width)
        {
            int up = (lr - 1) * TILE_SIZE;
            int mid = lr * TILE_SIZE;
            int down = (lr + 1) * TILE_SIZE;

            int fc0 = lc - 1;
            int fc1 = lc;
            int fc2 = lc + 1;

            float f0 = filter[0], f1 = filter[1], f2 = filter[2];
            float f3 = filter[3], f4 = filter[4], f5 = filter[5];
            float f6 = filter[6], f7 = filter[7], f8 = filter[8];

            float sum;

            sum = local_tile[up + fc0] * f0;
            sum = fma(local_tile[up + fc1], f1, sum);
            sum = fma(local_tile[up + fc2], f2, sum);

            sum = fma(local_tile[mid + fc0], f3, sum);
            sum = fma(local_tile[mid + fc1], f4, sum);
            sum = fma(local_tile[mid + fc2], f5, sum);

            sum = fma(local_tile[down + fc0], f6, sum);
            sum = fma(local_tile[down + fc1], f7, sum);
            sum = fma(local_tile[down + fc2], f8, sum);

            output_image[out_r * image_width + out_c] = sum;
        }
    }
}
