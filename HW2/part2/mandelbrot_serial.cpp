#include <immintrin.h>
/*

  Note: This code was modified from example code
  originally provided by Intel.  To comply with Intel's open source
  licensing agreement, their copyright is retained below.

  -----------------------------------------------------------------

  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

namespace
{

int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
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

} // namespace

//
// MandelbrotSerial --
//
// Compute an image visualizing the mandelbrot set.  The resulting
// array contains the number of iterations required before the complex
// number corresponding to a pixel could be rejected from the set.
//
// * x0, y0, x1, y1 describe the complex coordinates mapping
//   into the image viewport.
// * width, height describe the size of the output image
// * startRow, totalRows describe how much of the image to compute
void mandelbrot_serial(float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int start_row,
                       int total_rows,
                       int max_iterations,
                       int *output)
{
    float dx = (x1 - x0) / (float)width;
    float dy = (y1 - y0) / (float)height;

    int end_row = start_row + total_rows;

    for (int j = start_row; j < end_row; j++)
    {
        for (int i = 0; i < width; ++i)
        {
            float x = x0 + ((float)i * dx);
            float y = y0 + ((float)j * dy);

            int index = ((j * width) + i);
            output[index] = mandel(x, y, max_iterations);
        }
    }
}

void mandelbrot_avx2(float x0,
                     float y0,
                     float x1,
                     float y1,
                     int width,
                     int height,
                     int start_row,
                     int total_rows,
                     int max_iterations,
                     int *output)
{
    float dx = (x1 - x0) / (float)width;
    float dy = (y1 - y0) / (float)height;

    int end_row = start_row + total_rows;

    for (int j = start_row; j < end_row; j++)
    {
        __m256 y_vec = _mm256_set1_ps(y0 + j * dy);

        for (int i = 0; i < width; i += 8)
        {
            float x_vals[8];
            for (int k = 0; k < 8; ++k)
                x_vals[k] = x0 + (i + k) * dx;
            __m256 x_vec = _mm256_loadu_ps(x_vals);

            __m256 z_re = x_vec, c_re = x_vec;
            __m256 z_im = y_vec, c_im = y_vec;

            __m256i iter = _mm256_setzero_si256();
            __m256 four = _mm256_set1_ps(4.f);

            for (int n = 0; n < max_iterations; ++n)
            {
                __m256 z_re2 = _mm256_mul_ps(z_re, z_re);
                __m256 z_im2 = _mm256_mul_ps(z_im, z_im);
                __m256 mag2 = _mm256_add_ps(z_re2, z_im2);
                __m256 mask = _mm256_cmp_ps(mag2, four, _CMP_LE_OQ);
                if (_mm256_testz_ps(mask, _mm256_castsi256_ps(_mm256_set1_epi32(-1))))
                    break;
                
                __m256 new_re = _mm256_sub_ps(z_re2, z_im2);
                __m256 new_im = _mm256_mul_ps(_mm256_mul_ps(z_re, z_im), _mm256_set1_ps(2.f));
                z_re = _mm256_add_ps(c_re, new_re);
                z_im = _mm256_add_ps(c_im, new_im);
                __m256i mask_int = _mm256_castps_si256(mask);
                iter = _mm256_add_epi32(iter, _mm256_and_si256(mask_int, _mm256_set1_epi32(1)));
            }
            int index = j * width + i;
            int iter_vals[8];
            _mm256_storeu_si256((__m256i *)iter_vals, iter);

            // handle the remaining pixels
            int remaining = (8 > width - i)? (width - i) : 8;
            for (int k = 0; k < remaining; ++k)
                output[index + k] = iter_vals[k];
        }
    }
}