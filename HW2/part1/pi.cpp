#include <immintrin.h>
#include <random>
#include <thread>
#include <vector>

using namespace std;

struct xoshiro256_state
{
    __m256i s0, s1, s2, s3;

    void seed(uint64_t base_seed)
    {
        uint64_t seeds[32];
        uint64_t z = base_seed;
        for (int i = 0; i < 32; i++)
        {
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z ^= z >> 31;
            seeds[i] = z;
        }
        s0 = _mm256_loadu_si256((__m256i *)(seeds + 0));
        s1 = _mm256_loadu_si256((__m256i *)(seeds + 8));
        s2 = _mm256_loadu_si256((__m256i *)(seeds + 16));
        s3 = _mm256_loadu_si256((__m256i *)(seeds + 24));
    }

    __m256 randf()
    {
        __m256i t = _mm256_slli_epi64(s1, 17);

        s2 = _mm256_xor_si256(s2, s0);
        s3 = _mm256_xor_si256(s3, s1);
        s1 = _mm256_xor_si256(s1, s2);
        s0 = _mm256_xor_si256(s0, s3);
        s2 = _mm256_xor_si256(s2, t);
        s3 = _mm256_or_si256(_mm256_slli_epi64(s3, 45), _mm256_srli_epi64(s3, 64 - 45));

        // Convert s1 (or s0) to float in [0,1)
        __m256i res = s1; // take s1 as result
        __m256 resf = _mm256_cvtepi32_ps(_mm256_and_si256(res, _mm256_set1_epi32(0xFFFFFF)));
        return _mm256_mul_ps(resf, _mm256_set1_ps(1.0f / (1 << 24)));
    }
};

struct MonteCarloArgs
{
    uint64_t start;
    uint64_t chunk;
    uint64_t hits;
};

static void monte_carlo_thread(MonteCarloArgs *args)
{
    xoshiro256_state rng;
    rng.seed((uint64_t)random_device{}() + args->start);

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
    uint64_t start = 0;

    for (uint8_t t = 0; t < num_threads; ++t)
    {
        threadArgs[t] = {start, chunk, 0};
        start += chunk;
        threads[t] = thread(monte_carlo_thread, &threadArgs[t]);
    }

    uint64_t total_hits = 0;
    for (uint8_t t = 0; t < num_threads; ++t)
    {
        threads[t].join();
        total_hits += threadArgs[t].hits;
    }

    float pi = 4.0 * total_hits / n;
    printf("%f\n", pi);
    return 0;
}
