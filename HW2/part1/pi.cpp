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
    int t_id;
};

void monte_carlo_thread(MonteCarloArgs *args)
{
    xorshift64_state rng;
    rng.seed(args->t_id+77777);

    uint64_t local_hits = 0;
    uint64_t chunk = args->chunk;

    __m256 one = _mm256_set1_ps(1.0f);

    for (uint64_t i = 0; i < chunk / 8; i++)
    {
        __m256 x = rng.randf();
        __m256 y = rng.randf();

        __m256 sum = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));
        __m256 mask = _mm256_cmp_ps(sum, one, _CMP_LE_OS);
        local_hits += _mm_popcnt_u32(_mm256_movemask_ps(mask));
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
        threadArgs[t] = {chunk, 0, t};
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
