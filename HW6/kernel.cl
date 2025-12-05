// In this assignment, you will need to implement a GPU kernel function for convolution in OpenCL by using the zero-padding method. A serial implementation of convolution can be found in serial_conv() in serial_conv.c. You can refer to the implementation to port it to OpenCL. You may refer to this article to learn about the zero-padding method. Figure 2 shows an example of applying the zero-padding method to the source image (on the left) and thereby resulting a same-size, filtered output image (on the right).

// Tiled convolution kernel using __local (shared) memory.
// Each work-group computes a BLOCK_SIZE x BLOCK_SIZE block of output pixels.
// The local tile has extra halo = filter_radius on each side.
#define BLOCK_SIZE 16

__kernel void convolution(const int filter_width,
						  __global const float *filter,
						  const int image_height,
						  const int image_width,
						  __global const float *input_image,
						  __global float *output_image,
						  __local float *local_tile)
{
	const int R = filter_width / 2; // filter radius
	const int tile_size = BLOCK_SIZE + 2 * R;

	// Global coordinates of this work-item (output pixel)
	int global_row = get_global_id(0);
	int global_col = get_global_id(1);

	// Local coordinates within the work-group
	int local_row = get_local_id(0);
	int local_col = get_local_id(1);

	// Work-group origin (top-left of the BLOCK_SIZE block in global coordinates)
	int group_row = get_group_id(0) * BLOCK_SIZE;
	int group_col = get_group_id(1) * BLOCK_SIZE;

	const int local_nrows = get_local_size(0);
	const int local_ncols = get_local_size(1);

	// Each work-item cooperatively loads the tile (including halo) into local memory.
	for (int y = local_row; y < tile_size; y += local_nrows)
	{
		for (int x = local_col; x < tile_size; x += local_ncols)
		{
			int src_r = group_row + y - R;
			int src_c = group_col + x - R;
			float val = 0.0f;
			if (src_r >= 0 && src_r < image_height && src_c >= 0 && src_c < image_width)
			{
				val = input_image[src_r * image_width + src_c];
			}
			local_tile[y * tile_size + x] = val;
		}
	}

	// Wait for all loads to complete
	barrier(CLK_LOCAL_MEM_FENCE);

	// Only compute for valid output pixels
	if (global_row < image_height && global_col < image_width)
	{
		float sum = 0.0f;

		// Position of this pixel within the local_tile (offset by halo)
		int center_y = local_row + R;
		int center_x = local_col + R;

		for (int fy = -R; fy <= R; ++fy)
		{
			for (int fx = -R; fx <= R; ++fx)
			{
				float in_val = local_tile[(center_y + fy) * tile_size + (center_x + fx)];
				float fval = filter[(fy + R) * filter_width + (fx + R)];
				sum += in_val * fval;
			}
		}

		output_image[global_row * image_width + global_col] = sum;
	}

	// No need for further synchronization
}
