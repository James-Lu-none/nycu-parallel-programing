#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include "cycle_timer.h"

float g_x0, g_y0, g_x1, g_y1;
int g_width, g_height, g_max_iterations, g_num_threads;
int *g_output;

extern void mandelbrot_serial(float x0,
                              float y0,
                              float x1,
                              float y1,
                              int width,
                              int height,
                              int start_row,
                              int num_rows,
                              int max_iterations,
                              int *output);

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

    for (int row = threadId; row < g_height; row += g_num_threads)
    {
        mandelbrot_serial(g_x0, g_y0, g_x1, g_y1, g_width, g_height, row, 1, g_max_iterations, g_output);
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
    g_x0 = x0;
    g_y0 = y0;
    g_x1 = x1;
    g_y1 = y1;
    g_width = width;
    g_height = height;
    g_max_iterations = max_iterations;
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
