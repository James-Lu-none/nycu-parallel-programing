#include <cmath>
#include <immintrin.h>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "rng.hpp"

using namespace std;
constexpr double PI = 3.14159265358979323846;

template <typename RNG> double monte_carlo_pi_single(uint64_t samples, uint64_t seed)
{
    RNG rng;
    rng.seed(seed);
    uint64_t hits = 0;
    float buffer[8];
    float buffer2[8];
    uint64_t simd_iters = samples / 8;

    for (uint64_t i = 0; i < simd_iters; i++)
    {
        _mm256_storeu_ps(buffer, rng.randf());
        _mm256_storeu_ps(buffer2, rng.randf());
        for (uint64_t j = 0; j < 8; j++)
        {
            float x = buffer[j];
            float y = buffer2[j];
            if (x * x + y * y <= 1.0f)
                hits++;
        }
    }
    return hits;
}

template <typename RNG> double monte_carlo_pi(uint64_t total_samples, uint64_t num_threads)
{
    vector<thread> threads(num_threads);
    vector<double> results(num_threads, 0.0);

    for (uint64_t t = 0; t < num_threads; t++)
    {
        threads[t] = thread(
            [t, &results, total_samples, num_threads]() {
                double pi_est = monte_carlo_pi_single<RNG>(total_samples / num_threads, (uint64_t)random_device{}() + t);
                results[t] = pi_est;
            });
    }

    for (auto &th : threads)
        th.join();
    
    double sum = accumulate(results.begin(), results.end(), 0.0);
    return (4.0 * sum) / total_samples;
}

// Statistical test
template <typename RNG>
void monte_carlo_pi_stat(uint64_t total_samples, uint64_t num_threads, const string &name)
{
    using namespace std::chrono;

    auto start = high_resolution_clock::now();

    double estimate = monte_carlo_pi<RNG>(total_samples, num_threads);

    auto end = high_resolution_clock::now();
    double elapsed_sec = duration<double>(end - start).count();

    printf("%s: pi = %f, Error = %f Time = %.6f sec\n", name.c_str(), estimate, estimate - PI, elapsed_sec);
}

int main()
{
    const uint64_t num_samples = 5000000000;
    const uint64_t num_threads = 8;

    monte_carlo_pi_stat<xoshiro256_state>(num_samples, num_threads, "xoshiro256");
    monte_carlo_pi_stat<xoshiro64_state>(num_samples, num_threads, "xoshiro64");
    monte_carlo_pi_stat<xorshift64_state>(num_samples, num_threads, "xorshift64");
    return 0;
}
