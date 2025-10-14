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
    // printf("Hello world from thread %d\n", args->threadId);

    long threadId = (long) args;
    // double start_time = CycleTimer::current_seconds();
    for (int row = threadId; row < g_height; row += g_num_threads)
    {
        float y = g_y0 + row * dy;
        for (int col = 0; col < g_width; ++col)
        {
            float x = g_x0 + col * dx;

            // Check if in main cardioid
            float q = (x - 0.25f) * (x - 0.25f) + y * y;
            if (q * (q + (x - 0.25f)) <= 0.25f * y * y)
            {
                g_output[row * g_width + col] = 256;
                continue;
            }

            // Check if in period-2 bulb (main disk)
            if ((x + 1.0f) * (x + 1.0f) + y * y <= 0.0625f)
            {
                g_output[row * g_width + col] = 256;
                continue;
            }

            float z_re = x, z_im = y;
            int iter = 0;
            while (z_re * z_re + z_im * z_im <= 4.0f && iter < 256)
            {
                float new_re = z_re * z_re - z_im * z_im + x;
                float new_im = 2.0f * z_re * z_im + y;
                z_re = new_re;
                z_im = new_im;
                ++iter;
            }
            g_output[row * g_width + col] = iter;
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
