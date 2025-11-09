#include "cycle_timer.h"
#include <thread>
#include <vector>
#include <immintrin.h>

float g_x0, g_y0, dx, dy;
int g_width, g_height, g_num_threads;
int *g_output;

//
// worker_thread_start --
//
// Thread entrypoint.

void worker_thread_start(void *args)
{
    // printf("Hello world from thread %d\n", args->threadId);

    long threadId = (long) args;
    // double start_time = CycleTimer::current_seconds();
    
    __m128 _1 = _mm_set1_ps(1.f);
    __m128 _2 = _mm_set1_ps(2.f);
    __m128 _4 = _mm_set1_ps(4.f);
    __m128 _025 = _mm_set1_ps(0.25f);
    __m128 _00625 = _mm_set1_ps(0.0625f);
    
    for (int row = threadId; row < g_height; row += g_num_threads)
    {
        __m128 y_vec = _mm_set1_ps(g_y0 + row * dy);

        for (int i = 0; i < g_width; i += 4)
        {
            float x_vals[4];
            for (int k = 0; k < 4; ++k)
                x_vals[k] = g_x0 + (i + k) * dx;
            __m128 x_vec = _mm_loadu_ps(x_vals);

            // check if c=x+yi is in the main cardioid: q*(q+(x-0.25)) <= 0.25*y^2, q=(x-0.25f)^2+y^2
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
            if (inside_mask == 4){
                iter = _mm_set1_epi32(256);
            }
            else {
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
                    g_output[row * g_width + i + k] = 256;
                else
                    g_output[row * g_width + i + k] = iter_vals[k];
            }
        }
    }
    // double end_time = CycleTimer::current_seconds();
    // printf("Thread %d elapsed time: %lf seconds\n", threadId, end_time - start_time);
}

//
// mandelbrot_thread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrot_thread(int num_threads,
                       float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int max_iterations,
                       int *output)
{
    static constexpr int max_threads = 32;
    dx = (x1 - x0) / width;
    dy = (y1 - y0) / height;
    g_x0 = x0;
    g_y0 = y0;
    g_width = width;
    g_height = height;
    g_output = output;
    g_num_threads = num_threads;

    if (g_num_threads > max_threads)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", max_threads);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    // std::array<std::thread, > workers;
    std::vector<std::thread> workers(g_num_threads - 1);
    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 0; i < g_num_threads - 1; i++)
    {
        workers[i] = std::thread(worker_thread_start, (void *)(long)i);
    }

    worker_thread_start((void *)(long)(g_num_threads - 1));

    // join worker threads
    for (int i = 0; i < g_num_threads - 1; i++)
    {
        workers[i].join();
    }
}
