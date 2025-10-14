#include <immintrin.h>
#include <thread>
#include <vector>
#include "cycle_timer.h"

float g_x0, g_y0, dx, dy;
int g_width, g_height, g_num_threads;
int *g_output;

//
// worker_thread_start --
//
// Thread entrypoint.

void worker_thread_start(void *args)
{

    // TODO FOR PP STUDENTS: Implement the body of the worker
    // thread here. Each thread could make a call to mandelbrot_serial()
    // to compute a part of the output image. For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    // Of course, you can copy mandelbrot_serial() to this file and
    // modify it to pursue a better performance.

    // printf("Hello world from thread %d\n", args->threadId);

    long threadId = (long) args;
    // double start_time = CycleTimer::current_seconds();
    // mandelbrot_serial(g_x0, g_y0, g_x1, g_y1, g_width, g_height, row, step, g_max_iterations,
    //                   g_output);
    for (int row = threadId; row < g_height; row += g_num_threads)
    {
        __m256 y_vec = _mm256_set1_ps(g_y0 + row * dy);

        for (int i = 0; i < g_width; i += 8)
        {
            float x_vals[8];
            for (int k = 0; k < 8; ++k)
                x_vals[k] = g_x0 + (i + k) * dx;
            __m256 x_vec = _mm256_loadu_ps(x_vals);

            __m256 z_re = x_vec, c_re = x_vec;
            __m256 z_im = y_vec, c_im = y_vec;

            __m256i iter = _mm256_setzero_si256();
            __m256 four = _mm256_set1_ps(4.f);

            for (int n = 0; n < 256; ++n)
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
            int index = row * g_width + i;
            int iter_vals[8];
            _mm256_storeu_si256((__m256i *)iter_vals, iter);

            int remaining = (8 > g_width - i) ? (g_width - i) : 8;
            for (int k = 0; k < remaining; ++k)
                g_output[index + k] = iter_vals[k];
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
