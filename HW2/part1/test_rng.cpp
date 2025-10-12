#include <immintrin.h>
#include <random>
#include <thread>
#include <vector>
#include <chrono>
#include <iostream>

#include "rng.hpp"

using namespace std;

#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
using namespace std;

template <typename RNG> void benchmark_rng(const string &name, int num_samples, int num_threads)
{
    vector<thread> threads(num_threads);
    vector<double> results(num_threads, 0.0);

    auto start_time = chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; t++)
    {
        threads[t] = thread(
            [t, &results, num_samples, num_threads]()
            {
                RNG rng;
                rng.seed((uint64_t)random_device{}() + t);

                double local_sum = 0.0;
                int samples_per_thread = num_samples / num_threads;
                int simd_iters = samples_per_thread / 8;
                int leftover = samples_per_thread % 8;
                float buffer[8];

                for (int i = 0; i < simd_iters; i++)
                {
                    __m256 r = rng.randf();
                    _mm256_storeu_ps(buffer, r);
                    for (int j = 0; j < 8; j++)
                        local_sum += buffer[j];
                }

                for (int i = 0; i < leftover; i++)
                {
                    __m256 r = rng.randf();
                    _mm256_storeu_ps(buffer, r);
                    local_sum += buffer[0]; // use first value for leftover
                }

                results[t] = local_sum;
            });
    }

    for (auto &th : threads)
        th.join();

    double total_sum = 0.0;
    for (const auto &res : results)
        total_sum += res;

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end_time - start_time;

    cout << name << ": Sum = " << total_sum << ", Time = " << duration.count() << " seconds"
         << endl;
}

int main()
{
    const int num_samples = 100'000'000;
    const int num_threads = 1;

    benchmark_rng<xoshiro256_state>("xoshiro256", num_samples, num_threads);
    benchmark_rng<xoshiro64_state>("xoshiro64", num_samples, num_threads);
    benchmark_rng<xorshift64_state>("xorshift64", num_samples, num_threads);

    return 0;
}
