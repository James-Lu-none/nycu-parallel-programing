// Optimized tiled convolution kernel with loop unrolling
#define BLOCK_SIZE 32

static inline void load_tile(
    __global const float* input_image,
    __local float* local_tile,
    int image_height,
    int image_width,
    int group_row,
    int group_col,
    int R,
    int tile_size
) {
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int local_nrows = get_local_size(0);
    int local_ncols = get_local_size(1);

    for (int y = local_row; y < tile_size; y += local_nrows) {
        for (int x = local_col; x < tile_size; x += local_ncols) {
            int src_r = group_row + y - R;
            int src_c = group_col + x - R;

            float val = 0.0f;
            if (src_r >= 0 && src_r < image_height &&
                src_c >= 0 && src_c < image_width) {
                val = input_image[src_r * image_width + src_c];
            }

            local_tile[y * tile_size + x] = val;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

// General purpose kernel with pragma unroll
__kernel void convolution(const int filter_width,
                           __constant const float *filter,
                          const int image_height,
                          const int image_width,
                          __global const float *input_image,
                          __global float *output_image,
                          __local float *local_tile)
{
    const int R = filter_width / 2;
    const int tile_size = BLOCK_SIZE + 2 * R;

    int global_row = get_global_id(0);
    int global_col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int group_row = get_group_id(0) * BLOCK_SIZE;
    int group_col = get_group_id(1) * BLOCK_SIZE;

    const int local_nrows = get_local_size(0);
    const int local_ncols = get_local_size(1);

    load_tile(
        input_image,
        local_tile,
        image_height,
        image_width,
        group_row,
        group_col,
        R,
        tile_size
    );
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_row < image_height && global_col < image_width)
    {
        float sum = 0.0f;
        int center_y = local_row + R;
        int center_x = local_col + R;

        // Fully unroll the filter convolution loop
        #pragma unroll
        for (int fy = -R; fy <= R; ++fy)
        {
            int tile_y = (center_y + fy) * tile_size;
            int filter_y = (fy + R) * filter_width;
            
            #pragma unroll
            for (int fx = -R; fx <= R; ++fx)
            {
                float in_val = local_tile[tile_y + (center_x + fx)];
                float fval = filter[filter_y + (fx + R)];
                sum += in_val * fval;
            }
        }

        output_image[global_row * image_width + global_col] = sum;
    }
}

// Specialized 3x3 filter (9 operations)
__kernel void convolution_3x3(const int filter_width,
                               __constant const float *filter,
                               const int image_height,
                               const int image_width,
                               __global const float *input_image,
                               __global float *output_image,
                               __local float *local_tile)
{
    const int R = 1;
    const int tile_size = BLOCK_SIZE + 2;

    int global_row = get_global_id(0);
    int global_col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int group_row = get_group_id(0) * BLOCK_SIZE;
    int group_col = get_group_id(1) * BLOCK_SIZE;

    load_tile(
        input_image,
        local_tile,
        image_height,
        image_width,
        group_row,
        group_col,
        R,
        tile_size
    );

    if (global_row < image_height && global_col < image_width)
    {
        int center_y = local_row + R;
        int center_x = local_col + R;

        // Manually unrolled 3x3
        float sum = 0.0f;
        sum += local_tile[(center_y-1)*tile_size + (center_x-1)] * filter[0];
        sum += local_tile[(center_y-1)*tile_size + (center_x  )] * filter[1];
        sum += local_tile[(center_y-1)*tile_size + (center_x+1)] * filter[2];
        sum += local_tile[(center_y  )*tile_size + (center_x-1)] * filter[3];
        sum += local_tile[(center_y  )*tile_size + (center_x  )] * filter[4];
        sum += local_tile[(center_y  )*tile_size + (center_x+1)] * filter[5];
        sum += local_tile[(center_y+1)*tile_size + (center_x-1)] * filter[6];
        sum += local_tile[(center_y+1)*tile_size + (center_x  )] * filter[7];
        sum += local_tile[(center_y+1)*tile_size + (center_x+1)] * filter[8];

        output_image[global_row * image_width + global_col] = sum;
    }
}

// Specialized 5x5 filter (25 operations)
__kernel void convolution_5x5(const int filter_width,
                               __constant const float *filter,
                               const int image_height,
                               const int image_width,
                               __global const float *input_image,
                               __global float *output_image,
                               __local float *local_tile)
{
    const int R = 2;
    const int tile_size = BLOCK_SIZE + 4;

    int global_row = get_global_id(0);
    int global_col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int group_row = get_group_id(0) * BLOCK_SIZE;
    int group_col = get_group_id(1) * BLOCK_SIZE;

    load_tile(
        input_image,
        local_tile,
        image_height,
        image_width,
        group_row,
        group_col,
        R,
        tile_size
    );

    if (global_row < image_height && global_col < image_width)
    {
        int center_y = local_row + R;
        int center_x = local_col + R;
        float sum = 0.0f;

        // Row -2
        sum += local_tile[(center_y-2)*tile_size + (center_x-2)] * filter[0];
        sum += local_tile[(center_y-2)*tile_size + (center_x-1)] * filter[1];
        sum += local_tile[(center_y-2)*tile_size + (center_x  )] * filter[2];
        sum += local_tile[(center_y-2)*tile_size + (center_x+1)] * filter[3];
        sum += local_tile[(center_y-2)*tile_size + (center_x+2)] * filter[4];
        // Row -1
        sum += local_tile[(center_y-1)*tile_size + (center_x-2)] * filter[5];
        sum += local_tile[(center_y-1)*tile_size + (center_x-1)] * filter[6];
        sum += local_tile[(center_y-1)*tile_size + (center_x  )] * filter[7];
        sum += local_tile[(center_y-1)*tile_size + (center_x+1)] * filter[8];
        sum += local_tile[(center_y-1)*tile_size + (center_x+2)] * filter[9];
        // Row 0
        sum += local_tile[(center_y  )*tile_size + (center_x-2)] * filter[10];
        sum += local_tile[(center_y  )*tile_size + (center_x-1)] * filter[11];
        sum += local_tile[(center_y  )*tile_size + (center_x  )] * filter[12];
        sum += local_tile[(center_y  )*tile_size + (center_x+1)] * filter[13];
        sum += local_tile[(center_y  )*tile_size + (center_x+2)] * filter[14];
        // Row +1
        sum += local_tile[(center_y+1)*tile_size + (center_x-2)] * filter[15];
        sum += local_tile[(center_y+1)*tile_size + (center_x-1)] * filter[16];
        sum += local_tile[(center_y+1)*tile_size + (center_x  )] * filter[17];
        sum += local_tile[(center_y+1)*tile_size + (center_x+1)] * filter[18];
        sum += local_tile[(center_y+1)*tile_size + (center_x+2)] * filter[19];
        // Row +2
        sum += local_tile[(center_y+2)*tile_size + (center_x-2)] * filter[20];
        sum += local_tile[(center_y+2)*tile_size + (center_x-1)] * filter[21];
        sum += local_tile[(center_y+2)*tile_size + (center_x  )] * filter[22];
        sum += local_tile[(center_y+2)*tile_size + (center_x+1)] * filter[23];
        sum += local_tile[(center_y+2)*tile_size + (center_x+2)] * filter[24];

        output_image[global_row * image_width + global_col] = sum;
    }
}

// Specialized 7x7 filter (49 operations)
__kernel void convolution_7x7(const int filter_width,
                               __constant const float *filter,
                               const int image_height,
                               const int image_width,
                               __global const float *input_image,
                               __global float *output_image,
                               __local float *local_tile)
{
    const int R = 3;
    const int tile_size = BLOCK_SIZE + 6;

    int global_row = get_global_id(0);
    int global_col = get_global_id(1);
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int group_row = get_group_id(0) * BLOCK_SIZE;
    int group_col = get_group_id(1) * BLOCK_SIZE;

    load_tile(
        input_image,
        local_tile,
        image_height,
        image_width,
        group_row,
        group_col,
        R,
        tile_size
    );

    if (global_row < image_height && global_col < image_width)
    {
        int center_y = local_row + R;
        int center_x = local_col + R;
        float sum = 0.0f;

        // Row -3
        sum += local_tile[(center_y-3)*tile_size + (center_x-3)] * filter[0];
        sum += local_tile[(center_y-3)*tile_size + (center_x-2)] * filter[1];
        sum += local_tile[(center_y-3)*tile_size + (center_x-1)] * filter[2];
        sum += local_tile[(center_y-3)*tile_size + (center_x  )] * filter[3];
        sum += local_tile[(center_y-3)*tile_size + (center_x+1)] * filter[4];
        sum += local_tile[(center_y-3)*tile_size + (center_x+2)] * filter[5];
        sum += local_tile[(center_y-3)*tile_size + (center_x+3)] * filter[6];
        
        // Row -2
        sum += local_tile[(center_y-2)*tile_size + (center_x-3)] * filter[7];
        sum += local_tile[(center_y-2)*tile_size + (center_x-2)] * filter[8];
        sum += local_tile[(center_y-2)*tile_size + (center_x-1)] * filter[9];
        sum += local_tile[(center_y-2)*tile_size + (center_x  )] * filter[10];
        sum += local_tile[(center_y-2)*tile_size + (center_x+1)] * filter[11];
        sum += local_tile[(center_y-2)*tile_size + (center_x+2)] * filter[12];
        sum += local_tile[(center_y-2)*tile_size + (center_x+3)] * filter[13];
        
        // Row -1
        sum += local_tile[(center_y-1)*tile_size + (center_x-3)] * filter[14];
        sum += local_tile[(center_y-1)*tile_size + (center_x-2)] * filter[15];
        sum += local_tile[(center_y-1)*tile_size + (center_x-1)] * filter[16];
        sum += local_tile[(center_y-1)*tile_size + (center_x  )] * filter[17];
        sum += local_tile[(center_y-1)*tile_size + (center_x+1)] * filter[18];
        sum += local_tile[(center_y-1)*tile_size + (center_x+2)] * filter[19];
        sum += local_tile[(center_y-1)*tile_size + (center_x+3)] * filter[20];
        
        // Row 0
        sum += local_tile[(center_y  )*tile_size + (center_x-3)] * filter[21];
        sum += local_tile[(center_y  )*tile_size + (center_x-2)] * filter[22];
        sum += local_tile[(center_y  )*tile_size + (center_x-1)] * filter[23];
        sum += local_tile[(center_y  )*tile_size + (center_x  )] * filter[24];
        sum += local_tile[(center_y  )*tile_size + (center_x+1)] * filter[25];
        sum += local_tile[(center_y  )*tile_size + (center_x+2)] * filter[26];
        sum += local_tile[(center_y  )*tile_size + (center_x+3)] * filter[27];
        
        // Row +1
        sum += local_tile[(center_y+1)*tile_size + (center_x-3)] * filter[28];
        sum += local_tile[(center_y+1)*tile_size + (center_x-2)] * filter[29];
        sum += local_tile[(center_y+1)*tile_size + (center_x-1)] * filter[30];
        sum += local_tile[(center_y+1)*tile_size + (center_x  )] * filter[31];
        sum += local_tile[(center_y+1)*tile_size + (center_x+1)] * filter[32];
        sum += local_tile[(center_y+1)*tile_size + (center_x+2)] * filter[33];
        sum += local_tile[(center_y+1)*tile_size + (center_x+3)] * filter[34];
        
        // Row +2
        sum += local_tile[(center_y+2)*tile_size + (center_x-3)] * filter[35];
        sum += local_tile[(center_y+2)*tile_size + (center_x-2)] * filter[36];
        sum += local_tile[(center_y+2)*tile_size + (center_x-1)] * filter[37];
        sum += local_tile[(center_y+2)*tile_size + (center_x  )] * filter[38];
        sum += local_tile[(center_y+2)*tile_size + (center_x+1)] * filter[39];
        sum += local_tile[(center_y+2)*tile_size + (center_x+2)] * filter[40];
        sum += local_tile[(center_y+2)*tile_size + (center_x+3)] * filter[41];
        
        // Row +3
        sum += local_tile[(center_y+3)*tile_size + (center_x-3)] * filter[42];
        sum += local_tile[(center_y+3)*tile_size + (center_x-2)] * filter[43];
        sum += local_tile[(center_y+3)*tile_size + (center_x-1)] * filter[44];
        sum += local_tile[(center_y+3)*tile_size + (center_x  )] * filter[45];
        sum += local_tile[(center_y+3)*tile_size + (center_x+1)] * filter[46];
        sum += local_tile[(center_y+3)*tile_size + (center_x+2)] * filter[47];
        sum += local_tile[(center_y+3)*tile_size + (center_x+3)] * filter[48];

        output_image[global_row * image_width + global_col] = sum;
    }
}