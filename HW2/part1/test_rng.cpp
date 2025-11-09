#include <cmath>
#include <immintrin.h>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "rng.hpp"

using namespace std;

template <typename RNG>
void statistical_test(uint64_t total_samples, uint64_t num_threads, const string &name)
{
    printf("==================================\n");
    printf("Testing RNG: %s with %lu samples using %lu threads\n", name.c_str(), total_samples, num_threads);
    const int num_bins = 10;
    vector<int> bins(num_bins, 0);
    vector<thread> threads(num_threads);
    vector<vector<int>> thread_bins(num_threads, vector<int>(num_bins, 0));
    
    auto start = std::chrono::high_resolution_clock::now();

    for (uint64_t t = 0; t < num_threads; t++)
    {
        threads[t] = thread(
            [t, &thread_bins, total_samples, num_threads]() {
                RNG rng;
                rng.seed((uint64_t)random_device{}() + t);
                uint64_t samples_per_thread = total_samples / num_threads;
                float buffer[8];
                uint64_t simd_iters = samples_per_thread / 8;

                for (uint64_t i = 0; i < simd_iters; i++)
                {
                    _mm256_storeu_ps(buffer, rng.randf());
                    for (uint64_t j = 0; j < 8; j++)
                    {
                        if (i == 0)
                            printf("Thread %lu: %f\n", t, buffer[j]);
                        int bin_index = static_cast<int>(buffer[j] * num_bins);
                        if (bin_index == num_bins)
                            bin_index = num_bins - 1; // Edge case for value == 1.0
                        thread_bins[t][bin_index]++;
                    }
                }
            });
    }

    for (auto &th : threads)
        th.join();

    // Combine results from all threads
    for (uint64_t t = 0; t < num_threads; t++)
    {
        for (int b = 0; b < num_bins; b++)
        {
            bins[b] += thread_bins[t][b];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    // Print results
    printf("Time = %.6f sec\n", elapsed_sec);
    for (int b = 0; b < num_bins; b++)
    {
        printf("Bin %d: Count = %d, percentage = %f\n", b, bins[b], bins[b] * 100.0 / total_samples);
    }
    printf("==================================\n");
    return;
}

int main()
{
    const uint64_t num_samples = 1000;
    const uint64_t num_threads = 4;

    statistical_test<xoshiro256_state>(num_samples, num_threads, "xoshiro256");
    statistical_test<xoshiro64_state>(num_samples, num_threads, "xoshiro64");
    statistical_test<xorshift64_state>(num_samples, num_threads, "xorshift64");
    return 0;
}
