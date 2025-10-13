#pragma once
#include <cstdint>
#include <immintrin.h>
#include <random>
struct real_distribution
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution;
    real_distribution() : distribution(0.0f, 1.0f) {}
    void seed(uint64_t s) { generator.seed(s); }
    __m256 randf()
    {
        alignas(32) float vals[8];
        for (int i = 0; i < 8; i++)
            vals[i] = distribution(generator);
        return _mm256_loadu_ps(vals);
    }
};

constexpr float ONE_OVER_2_24 = 1.0f / 16777216.0f;
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
        __m256i sum = _mm256_add_epi64(s0, s3);
        __m256i result = _mm256_add_epi64(
            _mm256_or_si256(_mm256_slli_epi64(sum, 23), _mm256_srli_epi64(sum, 64 - 23)), s0);
        
        __m256i t = _mm256_slli_epi64(s1, 17);
        s2 = _mm256_xor_si256(s2, s0);
        s3 = _mm256_xor_si256(s3, s1);
        s1 = _mm256_xor_si256(s1, s2);
        s0 = _mm256_xor_si256(s0, s3);
        s2 = _mm256_xor_si256(s2, t);
        s3 = _mm256_or_si256(_mm256_slli_epi64(s3, 45), _mm256_srli_epi64(s3, 64 - 45));

        __m256 float_vals = _mm256_cvtepi32_ps(_mm256_srli_epi32(result, 8));
        return _mm256_mul_ps(float_vals, _mm256_set1_ps(1.0f / (1 << 24)));
    }
};

struct xoshiro256d_state
{
    __m256i s0, s1, s2, s3;
    alignas(32) uint64_t tmp[4];

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

    __m256d randd()
    {
        __m256i t = _mm256_slli_epi64(s1, 17);

        s2 = _mm256_xor_si256(s2, s0);
        s3 = _mm256_xor_si256(s3, s1);
        s1 = _mm256_xor_si256(s1, s2);
        s0 = _mm256_xor_si256(s0, s3);
        s2 = _mm256_xor_si256(s2, t);
        s3 = _mm256_or_si256(_mm256_slli_epi64(s3, 45), _mm256_srli_epi64(s3, 64 - 45));

        __m256i x = _mm256_srli_epi64(s1, 11);
        _mm256_store_si256((__m256i *)tmp, x);
        __m256d resd = _mm256_set_pd((double)tmp[3], (double)tmp[2], (double)tmp[1], (double)tmp[0]);
        return _mm256_mul_pd(resd, _mm256_set1_pd(1.0 / (1ULL << 53)));
    }
};

struct xoshiro64_state
{
    __m256i s0, s1;

    void seed(uint64_t base_seed)
    {
        uint64_t seeds[8];
        uint64_t z = base_seed;
        for (int i = 0; i < 8; i++)
        {
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z ^= z >> 31;
            seeds[i] = z;
        }
        s0 = _mm256_loadu_si256((__m256i *)(seeds + 0));
        s1 = _mm256_loadu_si256((__m256i *)(seeds + 4));
    }

    __m256 randf()
    {
        __m256i t = _mm256_slli_epi64(s1, 13);

        s1 = _mm256_xor_si256(s1, s0);
        s0 = _mm256_xor_si256(s0, s1);
        s1 = _mm256_xor_si256(s1, t);
        s0 = _mm256_or_si256(_mm256_slli_epi64(s0, 17), _mm256_srli_epi64(s0, 64 - 17));

        __m256i res = s0;
        __m256 resf = _mm256_cvtepi32_ps(_mm256_and_si256(res, _mm256_set1_epi32(0xFFFFFF)));
        return _mm256_mul_ps(resf, _mm256_set1_ps(1.0f / (1 << 24)));
    }
};

struct xorshift64_state
{
    __m256i s;

    void seed(uint64_t base_seed)
    {
        uint64_t seeds[4];
        uint64_t z = base_seed;
        for (int i = 0; i < 4; i++)
        {
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z ^= z >> 31;
            seeds[i] = z;
        }
        s = _mm256_loadu_si256((__m256i *)(seeds));
    }

    __m256 randf()
    {
        __m256i x = s;
        x = _mm256_xor_si256(x, _mm256_slli_epi64(x, 13));
        x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 7));
        x = _mm256_xor_si256(x, _mm256_slli_epi64(x, 17));
        s = x;

        __m256i res = _mm256_and_si256(x, _mm256_set1_epi32(0xFFFFFF));
        __m256 resf = _mm256_cvtepi32_ps(res);
        return _mm256_mul_ps(resf, _mm256_set1_ps(1.0f / (1 << 24)));
    }
};
