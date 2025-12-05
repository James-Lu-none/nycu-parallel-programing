// In this assignment, you will need to implement a GPU kernel function for convolution in OpenCL by using the zero-padding method. A serial implementation of convolution can be found in serial_conv() in serial_conv.c. You can refer to the implementation to port it to OpenCL. You may refer to this article to learn about the zero-padding method. Figure 2 shows an example of applying the zero-padding method to the source image (on the left) and thereby resulting a same-size, filtered output image (on the right).

__kernel void convolution(const int filter_width,
						  __global const float *filter,
						  const int image_height,
						  const int image_width,
						  __global const float *input_image,
						  __global float *output_image)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	if (i >= image_height || j >= image_width)
		return; // safeguard for extra work-items

	int halffilter_size = filter_width / 2;
	float sum = 0.0f;

	for (int k = -halffilter_size; k <= halffilter_size; k++) {
		for (int l = -halffilter_size; l <= halffilter_size; l++) {
			int ii = i + k;
			int jj = j + l;

			// zero-padding: if outside image bounds, treat as 0 (skip)
			if (ii >= 0 && ii < image_height && jj >= 0 && jj < image_width) {
				float in_val = input_image[ii * image_width + jj];
				int fk = k + halffilter_size;
				int fl = l + halffilter_size;
				float fval = filter[fk * filter_width + fl];
				sum += in_val * fval;
			}
		}
	}

	output_image[i * image_width + j] = sum;
}
