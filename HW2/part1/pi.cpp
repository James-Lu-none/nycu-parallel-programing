#include <cmath>
#include <cstdio>
#include <pthread.h>
#include <stdint.h>

void usage(char *progname) {
    fprintf(stderr, "Usage: %s <threads> <n>\n", progname);
    fprintf(stderr, "  <threads> : number of threads to use (positive integer)\n");
    fprintf(stderr, "  <n>       : number of intervals (positive integer)\n");
}

long threads;
long long int n;
long long int hits;
pthread_mutex_t mutex;
#include <immintrin.h>

// 256-bit state (4x64)
typedef struct
{
    uint64_t s[4];
} xoshiro256_state;

static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro256ss(xoshiro256_state *state)
{
    const uint64_t result = rotl(state->s[1] * 5, 7) * 9;

    const uint64_t t = state->s[1] << 17;

    state->s[2] ^= state->s[0];
    state->s[3] ^= state->s[1];
    state->s[1] ^= state->s[2];
    state->s[0] ^= state->s[3];

    state->s[2] ^= t;
    state->s[3] = rotl(state->s[3], 45);

    return result;
}

#include <time.h>

void seed_xoshiro(xoshiro256_state *st, uint64_t seed)
{
    // Simple splitmix64 seeding
    uint64_t z = seed + 0x9E3779B97f4A7C15ULL;
    for (int i = 0; i < 4; i++)
    {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        st->s[i] = z;
    }
}

float randf(xoshiro256_state *st)
{
    return (xoshiro256ss(st) >> 40) * (1.0f / (1ULL << 24));
}

void *thread_monte_carlo_pi(void *arg)
{
    long threadId = (long)arg;
    xoshiro256_state rng;
    seed_xoshiro(&rng, time(NULL) ^ (threadId * 0x9E3779B1));
    int baseChunk = n / threads;
    int remainder = n % threads;
    int start = threadId * baseChunk + (threadId < remainder ? threadId : remainder);
    int end = start + baseChunk + (threadId < remainder ? 1 : 0);
    long local_hits = 0;
    
    int simd_iters = (end - start) / 8;
    int leftover = (end - start) % 8;

    for (int i = 0; i < simd_iters; ++i) {
        float xs[8], ys[8];
        for (int j = 0; j < 8; ++j) {
            xs[j] = randf(&rng);
            ys[j] = randf(&rng);
        }
        __m256 vx = _mm256_loadu_ps(xs);
        __m256 vy = _mm256_loadu_ps(ys);
        __m256 vxsq = _mm256_mul_ps(vx, vx);
        __m256 vysq = _mm256_mul_ps(vy, vy);
        __m256 vsum = _mm256_add_ps(vxsq, vysq);
        __m256 vmask = _mm256_cmp_ps(vsum, _mm256_set1_ps(1.0f), _CMP_LE_OS);
        local_hits += _mm_popcnt_u32(_mm256_movemask_ps(vmask));
    }

    for (int i = 0; i < leftover; ++i) {
        float x = randf(&rng);
        float y = randf(&rng);
        if (x * x + y * y <= 1.0f) {
            local_hits++;
        }
    }

    pthread_mutex_lock(&mutex);
    hits += local_hits;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    char *endptr;
    threads = strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || threads <= 0) {
        fprintf(stderr, "Invalid value for threads: %s\n", argv[1]);
        usage(argv[0]);
        return 1;
    }

    n = strtoll(argv[2], &endptr, 10);
    if (*endptr != '\0') {
        fprintf(stderr, "Invalid value for n: %s\n", argv[2]);
        usage(argv[0]);
        return 1;
    }

    hits = 0;
    pthread_mutex_init(&mutex, NULL);

    pthread_t *thread_handles;
    thread_handles = (pthread_t *)malloc(threads * sizeof(pthread_t));

    for (long t=0; t<threads; ++t){
        pthread_create(&thread_handles[t], NULL, thread_monte_carlo_pi, (void*)t);
    }
    for (long t = 0; t < threads; ++t)
    {
        pthread_join(thread_handles[t], NULL);
    }

    double pi = hits * 4.0 / n;
    printf("%lf\n", pi);
    free(thread_handles);
    return 0;
}