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

    int k_start = -halffilter_size + i >= 0 ? -halffilter_size : 0;
    int k_end = halffilter_size + i < image_height ? halffilter_size : halffilter_size + i - image_height - 1;
    int l_start = -halffilter_size + j >= 0 ? -halffilter_size : 0;
    int l_end = halffilter_size + j < image_width ? halffilter_size : halffilter_size + j - image_width - 1;

    for (int k = k_start; k <= k_end; ++k) {
        for (int l = l_start; l <= l_end; ++l) {
            sum += input_image[(i + k) * image_width + j + l] * filter[(k + halffilter_size) * filter_width + l + halffilter_size];
        }
    }
	output_image[i * image_width + j] = sum;
}
