#include <algorithm>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <thread>

void TopDownMerge(double* a, double* aEnd, double* b, double* bEnd, double* dest)  
{  
    auto aStart = a;
    auto dstStart = dest;

    while (a < aEnd && b < bEnd)  
    {  
        if( *a < *b )  
            *dest++ = *a++;  
        else  
            *dest++ = *b++;  
    }
    
    while (a < aEnd)
    {
        *dest++ = *a++;
    }

    while (b < bEnd)
    {
        *dest++ = *b++;
    }

    std::copy(dstStart, dest, aStart);
}

void TopDownMergeInPlace(double* a, size_t begin, size_t mid, size_t end)
{
    size_t p1 = begin;
    size_t p2 = mid;
    while (p1 < mid && p2 < end)
    {
        if (a[p1] < a[p2])
        {
            ++p1;
        }
        else
        {
            auto temp = a[p2];
            size_t idx = p2;
            while (idx > p1)
            {
                a[idx] = a[idx - 1];
                --idx;
            }

            a[p1] = temp;
            ++p1;
            ++p2;
            ++mid;
        }
    }
}

#define SORT_AND_SWAP_2(b, i0, i1)\
    do { \
        auto tmp1 = b[i0]; \
        auto tmp2 = b[i1]; \
        auto cond = b[i0] > b[i1]; \
        b[i0] = cond ? tmp2 : tmp1; \
        b[i1] = cond ? tmp1 : tmp2; \
    } while (0)

// [[0 1][2 3][0 2][1 3][1 2]]
__attribute__((always_inline))
void sort4(double* b, size_t iBegin)
{
    auto bb = b + iBegin;

    // compare & swap [0 1][2 3]
    auto a0123 = _mm256_loadu_pd(bb);
    auto a1032 = _mm256_shuffle_pd(a0123, a0123, 1 + 4);

    auto min = _mm256_min_pd(a0123, a1032);
    auto max = _mm256_max_pd(a0123, a1032);
    auto res = _mm256_blend_pd(min, max, 8 + 2);

    // [0 2][1 3]
    auto a0213 = (__m256d)_mm256_permute4x64_epi64(res, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
    auto a2031 = _mm256_shuffle_pd(a0213, a0213, 1 + 4);

    min = _mm256_min_pd(a0213, a2031);
    max = _mm256_max_pd(a0213, a2031);
    res = _mm256_blend_pd(min, max, 8 + 2);

    // [1 2]
    auto a1203 = (__m256d)_mm256_permute4x64_epi64(res, (2 << 0) + (1 << 2) + (0 << 4) + (3 << 6));
    auto a2103 = _mm256_shuffle_pd(a1203, a1203, 1 + 0 + 8);

    // Move [1 2] to first lane for comparison purposes, then move it back.
    min = _mm256_min_pd(a1203, a2103);
    max = _mm256_max_pd(a1203, a2103);
    auto blended = _mm256_blend_pd(min, max, 8 + 2);

    res = (__m256d)_mm256_permute4x64_epi64(blended, (2 << 0) + (0 << 2) + (1 << 4) + (3 << 6));

    _mm256_storeu_pd(bb, res);
}

__attribute__((always_inline))
void sort3(double* b, size_t iBegin)
{
    auto bb = b + iBegin;
    SORT_AND_SWAP_2(bb, 0, 1);
    SORT_AND_SWAP_2(bb, 0, 2);
    SORT_AND_SWAP_2(bb, 1, 2);
}

__attribute__((always_inline))
void sort2(double* b, size_t iBegin)
{
    auto bb = b + iBegin;
    SORT_AND_SWAP_2(bb, 0, 1);
}

void TopDownSplitMerge(double* A, size_t iBegin, size_t iEnd, double* B)
{
    switch (iEnd - iBegin)
    {
        case 4:
        {
            return sort4(A, iBegin);    
        }
        case 3:
        {
            return sort3(A, iBegin);    
        }
        case 2:
        {
            return sort2(A, iBegin);    
        }
        case 0:
        case 1:
        {
            return;
        }
        default:
        {
            break;
        }
    }                             
    
    auto iMiddle = (iEnd + iBegin) / 2;              
    
    TopDownSplitMerge(A, iBegin, iMiddle, B);  
    TopDownSplitMerge(A, iMiddle, iEnd, B);  
    
    TopDownMerge(A+iBegin, A+iMiddle, A+iMiddle, A+iEnd, B+iBegin);
}

void TopDownMergeSort(double* A, double* B, size_t n)
{
    if (n < 1000)
    {
        return std::stable_sort(A, A + n);
    }

    const size_t thread_count = 16;
    std::thread threads[thread_count];

    const size_t n8 = thread_count * (n / thread_count);

    for (size_t i = 0; i < thread_count; ++i)
    {
        if (i != thread_count - 1)
        {
            threads[i] = std::thread([=] { TopDownSplitMerge(A, i * (n / thread_count), (i + 1) * (n / thread_count), B); });
        }
        else
        {
            TopDownSplitMerge(A, i * (n / thread_count), n, B);
        }
    }

    for (auto& t : threads) if (t.joinable()) t.join();

    size_t threads_merged = thread_count;
    while (threads_merged != 1)
    {
        for (size_t i = 0; i < threads_merged; i += 2)
        {
            if (i != threads_merged - 2)
            {
                threads[i / 2] = std::thread([=] { TopDownMerge(A + i * (n8 / threads_merged), A + (i + 1) * (n8 / threads_merged), A + (i + 1) * (n8 / threads_merged), A + (i + 2) * (n8 / threads_merged), B + i * (n8 / threads_merged)); });
            }
            else
            {
                TopDownMerge(A + i * (n8 / threads_merged), A + (i + 1) * (n8 / threads_merged), A + (i + 1) * (n8 / threads_merged), A + n, B + i * (n8 / threads_merged));
            }
        }

        for (auto& t : threads) if (t.joinable()) t.join();

        threads_merged /= 2;
    }
}

int main(int, char** argv)
{
    double* a = nullptr;
    double* b = nullptr;;
    auto f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    rewind(f);

    size_t len = sz / sizeof(double);

    a = (double*) malloc(sz);
    if (sz >= 1000)
    {
        b = (double*) malloc(sz);
    }
    
    fread(a, sz, 1, f);

    TopDownMergeSort(a, b, len);

    // for (size_t i = 0; i < len; ++i)
    // {
    //     printf("%g\n", a[i]);
    // }

    auto fout = fopen("output.bin", "wb");
    fwrite(a, sizeof(double), len, fout);
    fclose(fout);
}