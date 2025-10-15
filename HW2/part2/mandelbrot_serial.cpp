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
#include <immintrin.h>

namespace
{

int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    float q = (z_re - 0.25f) * (z_re - 0.25f) + z_im * z_im;
    if (q * (q + (z_re - 0.25f)) <= 0.25f * z_im * z_im)
    {
        return 256;
    }

    // check if c=x+yi is in the period-2 bulb (main disk)
    if ((z_re + 1.0f) * (z_re + 1.0f) + z_im * z_im <= 0.0625f)
    {
        return 256;
    }
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
    __m128 _1 = _mm_set1_ps(1.f);
    __m128 _2 = _mm_set1_ps(2.f);
    __m128 _4 = _mm_set1_ps(4.f);
    __m128 _025 = _mm_set1_ps(0.25f);
    __m128 _00625 = _mm_set1_ps(0.0625f);

    int end_row = start_row + total_rows;

    for (int j = start_row; j < end_row; j++)
    {
        __m128 y_vec = _mm_set1_ps(y0 + j * dy);
        for (int i = 0; i < width; i+=4)
        {
            float x_vals[4];
            for (int k = 0; k < 4; ++k)
                x_vals[k] = x0 + (i + k) * dx;
            __m128 x_vec = _mm_loadu_ps(x_vals);

            // check if c=x+yi is in the main cardioid: q*(q+(x-0.25)) <= 0.25*y^2,
            // q=(x-0.25f)^2+y^2
            __m128 x025 = _mm_sub_ps(x_vec, _025);
            __m128 y2 = _mm_mul_ps(y_vec, y_vec);
            __m128 q = _mm_add_ps(_mm_mul_ps(x025, x025), y2);
            __m128 left = _mm_mul_ps(q, _mm_add_ps(q, x025));
            __m128 right = _mm_mul_ps(_025, y2);
            __m128 mask_cardioid = _mm_cmple_ps(left, right);

            // check if c=x+yi is in the period-2 bulb (main disk): (x+1)^2+y^2 <= 0.0625
            __m128 xa1 = _mm_add_ps(x_vec, _1);
            __m128 mask_disk = _mm_cmple_ps(_mm_add_ps(_mm_mul_ps(xa1, xa1), y2), _00625);

            __m128 mask_inside = _mm_or_ps(mask_cardioid, mask_disk);
            int inside_mask = _mm_movemask_ps(mask_inside);

            __m128i iter = _mm_setzero_si128();

            // printf("row %d, col %d, inside_mask %d\n", row, i, inside_mask);
            if (inside_mask == 4)
            {
                iter = _mm_set1_epi32(256);
            }
            else
            {
                __m128 z_re = x_vec, c_re = x_vec;
                __m128 z_im = y_vec, c_im = y_vec;
                for (int n = 0; n < 256; ++n)
                {
                    __m128 z_re2 = _mm_mul_ps(z_re, z_re);
                    __m128 z_im2 = _mm_mul_ps(z_im, z_im);
                    __m128 mag2 = _mm_add_ps(z_re2, z_im2);

                    __m128 mask = _mm_cmple_ps(mag2, _4);
                    if (_mm_movemask_ps(mask) == 0)
                        break;

                    __m128 new_re = _mm_sub_ps(z_re2, z_im2);
                    __m128 new_im = _mm_mul_ps(_mm_mul_ps(z_re, z_im), _2);

                    z_re = _mm_add_ps(c_re, new_re);
                    z_im = _mm_add_ps(c_im, new_im);

                    __m128i mask_i = _mm_castps_si128(mask);
                    iter = _mm_add_epi32(iter, _mm_and_si128(mask_i, _mm_set1_epi32(1)));
                }
            }

            int iter_vals[4];
            _mm_storeu_si128((__m128i *)iter_vals, iter);

            for (int k = 0; k < 4; ++k)
            {
                if (inside_mask & (1 << k))
                    output[j * width + i + k] = 256;
                else
                    output[j * width + i + k] = iter_vals[k];
            }
        }
    }
}
