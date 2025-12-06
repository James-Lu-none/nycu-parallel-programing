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
    if (lr >= R && lr < R + BLOCK_SIZE &&
        lc >= R && lc < R + BLOCK_SIZE)
    {
        int out_r = group_row + (lr - R);
        int out_c = group_col + (lc - R);
        if (out_r < image_height && out_c < image_width)
        {
            float sum;
            // Manually unrolled loops for 3x3 filter
            sum = local_tile[(lr - 1) * TILE_SIZE + (lc - 1)] * filter[0];
            sum = fma(local_tile[(lr - 1) * TILE_SIZE + (lc    )], filter[1], sum);
            sum = fma(local_tile[(lr - 1) * TILE_SIZE + (lc + 1)], filter[2], sum);
            sum = fma(local_tile[(lr    ) * TILE_SIZE + (lc - 1)], filter[3], sum);
            sum = fma(local_tile[(lr    ) * TILE_SIZE + (lc    )], filter[4], sum);
            sum = fma(local_tile[(lr    ) * TILE_SIZE + (lc + 1)], filter[5], sum);
            sum = fma(local_tile[(lr + 1) * TILE_SIZE + (lc - 1)], filter[6], sum);
            sum = fma(local_tile[(lr + 1) * TILE_SIZE + (lc    )], filter[7], sum);
            sum = fma(local_tile[(lr + 1) * TILE_SIZE + (lc + 1)], filter[8], sum);
            output_image[out_r * image_width + out_c] = sum;
        }
    }
}