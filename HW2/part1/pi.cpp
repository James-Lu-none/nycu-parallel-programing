#include <immintrin.h>
#include <random>
#include <thread>
#include <vector>
#include "rng.hpp"

using namespace std;

struct MonteCarloArgs
{
    uint64_t chunk;
    uint64_t hits;
};

static void monte_carlo_thread(MonteCarloArgs *args)
{
    xoshiro256_state rng;

    rng.seed((uint64_t)random_device{}() | (uint64_t)random_device{}() << 32);

    uint64_t local_hits = 0;
    uint64_t chunk = args->chunk;

    __m256d one = _mm256_set1_pd(1.0);

    for (uint64_t i = 0; i < chunk / 8; i++)
    {
        __m256 x = rng.randf();
        __m256 y = rng.randf();
        // load 8 float to 4*2 double
        __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
        __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
        __m256d y_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(y));
        __m256d y_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(y, 1));

        __m256d sum_hi = _mm256_add_pd(_mm256_mul_pd(x_hi, x_hi), _mm256_mul_pd(y_hi, y_hi));
        __m256d mask_hi = _mm256_cmp_pd(sum_hi, one, _CMP_LE_OS);
        local_hits += _mm_popcnt_u32(_mm256_movemask_pd(mask_hi));
        __m256d sum_lo = _mm256_add_pd(_mm256_mul_pd(x_lo, x_lo), _mm256_mul_pd(y_lo, y_lo));
        __m256d mask_lo = _mm256_cmp_pd(sum_lo, one, _CMP_LE_OS);
        local_hits += _mm_popcnt_u32(_mm256_movemask_pd(mask_lo));
    }

    args->hits = local_hits;
}

static void monte_carlo_threadd(MonteCarloArgs *args)
{
    xoshiro256d_state rng;

    rng.seed((uint64_t)random_device{}() | (uint64_t)random_device{}() << 32);

    uint64_t local_hits = 0;
    uint64_t chunk = args->chunk;

    __m256d one = _mm256_set1_pd(1.0);

    for (uint64_t i = 0; i < chunk / 4; i++)
    {
        __m256d x = rng.randd();
        __m256d y = rng.randd();

        __m256d sum = _mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y));
        __m256d mask = _mm256_cmp_pd(sum, one, _CMP_LE_OS);
        local_hits += _mm_popcnt_u32(_mm256_movemask_pd(mask));
    }

    args->hits = local_hits;
}

void usage(char *progname)
{
    fprintf(stderr, "Usage: %s <threads> <n>\n", progname);
    fprintf(stderr, " <threads> : number of threads to use (positive integer)\n");
    fprintf(stderr, " <n> : number of intervals (positive integer)\n");
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        usage(argv[0]);
        return 1;
    }

    uint8_t num_threads;
    uint64_t n;
    char *endptr;
    num_threads = strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || num_threads <= 0)
    {
        fprintf(stderr, "Invalid value for threads: %s\n", argv[1]);
        usage(argv[0]);
        return 1;
    }
    n = strtoll(argv[2], &endptr, 10);
    if (*endptr != '\0' || n <= 0)
    {
        fprintf(stderr, "Invalid value for n: %s\n", argv[2]);
        usage(argv[0]);
        return 1;
    }

    vector<thread> threads(num_threads);
    vector<MonteCarloArgs> threadArgs(num_threads);

    uint64_t chunk = n / num_threads;

    for (uint8_t t = 0; t < num_threads; ++t)
    {
        threadArgs[t] = {chunk, 0};
        threads[t] = thread(monte_carlo_thread, &threadArgs[t]);
    }

    uint64_t total_hits = 0;
    for (uint8_t t = 0; t < num_threads; ++t)
    {
        threads[t].join();
        total_hits += threadArgs[t].hits;
    }

    double pi = 4.0 * total_hits / n;
    printf("%lf\n", pi);
    return 0;
}
