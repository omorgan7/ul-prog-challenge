#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>

// #define HYPERTHREADING
#ifdef HYPERTHREADING
constexpr int HYPERTHREAD_FACTOR = 1;
#else
constexpr int HYPERTHREAD_FACTOR = 1;
#endif

// lazy hack accounting for the fact that my laptop has 6 cores
// and the desktop has 8
#if __APPLE__
constexpr int THREAD_COUNT = 9 * HYPERTHREAD_FACTOR;
#else
constexpr int THREAD_COUNT = 12 * HYPERTHREAD_FACTOR;
#endif

#include <immintrin.h>

inline static __attribute__((always_inline)) __m256i
_mm256_mul_epi64(__m256i ymm3, __m256i ymm4) {
  // ported from GCC's implementation of a tight for loop.
  auto ymm0 = _mm256_srli_epi64(ymm3, 32);
  auto ymm2 = _mm256_srli_epi64(ymm4, 32);

  ymm0 = _mm256_mul_epu32(ymm0, ymm4);
  ymm2 = _mm256_mul_epu32(ymm2, ymm3);
  auto ymm1 = _mm256_mul_epu32(ymm3, ymm4);
  ymm0 = _mm256_add_epi64(ymm0, ymm2);
  ymm0 = _mm256_slli_epi64(ymm0, 32);
  ymm0 = _mm256_add_epi64(ymm1, ymm0);
  return ymm0;
}

#define CHECK_C_INT_CALL(call)                                                 \
  do {                                                                         \
    int ret = call;                                                            \
    if (ret < 0) {                                                             \
      perror(#call);                                                           \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

struct RowCol {
  uint32_t rows;
  uint32_t cols;
};

inline __attribute__((always_inline)) uint8_t *next_block(uint8_t *current,
                                                          uint8_t *end) {
  if (current == end) {
    return nullptr;
  }

  const RowCol &header = *reinterpret_cast<RowCol *>(current);
  return current + header.rows * header.cols * sizeof(int64_t) +
         sizeof(uint32_t) * 2;
}

inline __attribute__((always_inline)) int64_t *matrix_data(uint8_t *current) {
  return reinterpret_cast<int64_t *>(current + sizeof(uint32_t) * 2);
}

static inline __attribute__((always_inline)) void
matmul_noavx2(const RowCol &m0_header, const RowCol &m1_header,
              const RowCol &m2_header, int64_t *m0, int64_t *m1, int64_t *m2) {
  auto stride = m0_header.cols;
  auto stride4 = (stride / 4) * 4;

  for (uint32_t y = 0; y < m2_header.rows; ++y) {
    for (uint32_t x = 0; x < m2_header.cols; ++x) {
      int64_t acc = 0;
      for (uint32_t s = 0; s < stride; ++s) {
        acc += m0[y * m0_header.cols + s] * m1[x + s * m1_header.cols];
      }
      m2[m2_header.cols * y + x] = acc;
    }
  }
}

static inline __attribute__((always_inline)) void
matmul(const RowCol &m0_header, const RowCol &m1_header,
       const RowCol &m2_header, int64_t *m0, int64_t *m1, int64_t *m2,
       uint32_t row_end, uint32_t y = 0) {
  auto stride = m0_header.cols;
  auto stride4 = (stride / 4) * 4;

  auto cols = m2_header.cols;
  auto cols4 = (cols / 4) * 4;

  for (; y < row_end; ++y) {
    uint32_t x = 0;
    for (; x < cols4; x += 4) {
      __m256i acc = _mm256_setzero_si256();

      uint32_t s = 0;
      for (; s < stride4;) {
        auto x00 = _mm256_loadu_si256(
            reinterpret_cast<__m256i *>(m0 + y * m0_header.cols + s));

        __m256i x0, x1, shuf;
        x0 = _mm256_broadcastq_epi64(*reinterpret_cast<__m128i *>(&x00));
        x1 = _mm256_loadu_si256(
            reinterpret_cast<__m256i *>((m1 + x + s * m1_header.cols)));

        acc = _mm256_add_epi64(acc, _mm256_mul_epi64(x0, x1));
        ++s;

        // performance penalty for crossing lanes
        shuf =
            _mm256_permute4x64_epi64(x00, 1 + (1 << 2) + (2 << 4) + (3 << 6));
        x0 = _mm256_broadcastq_epi64(*reinterpret_cast<__m128i *>(&shuf));
        x1 = _mm256_loadu_si256(
            reinterpret_cast<__m256i *>((m1 + x + s * m1_header.cols)));

        acc = _mm256_add_epi64(acc, _mm256_mul_epi64(x0, x1));
        ++s;

        // only cross lanes 1 way, not both
        shuf = _mm256_permute2f128_si256(x00, x00, 1 + (1 << 4));
        x0 = _mm256_broadcastq_epi64(*reinterpret_cast<__m128i *>(&shuf));
        x1 = _mm256_loadu_si256(
            reinterpret_cast<__m256i *>((m1 + x + s * m1_header.cols)));

        acc = _mm256_add_epi64(acc, _mm256_mul_epi64(x0, x1));
        ++s;

        shuf =
            _mm256_permute4x64_epi64(shuf, 1 + (1 << 2) + (2 << 4) + (3 << 6));
        x0 = _mm256_broadcastq_epi64(*reinterpret_cast<__m128i *>(&shuf));
        x1 = _mm256_loadu_si256(
            reinterpret_cast<__m256i *>((m1 + x + s * m1_header.cols)));

        acc = _mm256_add_epi64(acc, _mm256_mul_epi64(x0, x1));
        ++s;
      }
      for (; s < stride; ++s) {
        auto x0 = _mm256_castpd_si256(_mm256_broadcast_sd(
            reinterpret_cast<double *>(m0 + y * m0_header.cols + s)));
        auto x1 = _mm256_loadu_si256(
            reinterpret_cast<__m256i *>((m1 + x + s * m1_header.cols)));
        acc = _mm256_add_epi64(acc, _mm256_mul_epi64(x0, x1));
      }
      _mm256_storeu_si256(
          reinterpret_cast<__m256i *>(m2 + m2_header.cols * y + x), acc);
    }

    for (; x < cols; ++x) {
      int64_t acc = 0;
      for (uint32_t s = 0; s < stride; ++s) {
        acc += m0[y * m0_header.cols + s] * m1[x + s * m1_header.cols];
      }
      m2[m2_header.cols * y + x] = acc;
    }
  }
}

static void print(const RowCol &h, int64_t *d) {
  printf("[");
  for (auto y = 0; y < h.rows; ++y) {
    printf("[");
    for (auto x = 0; x < h.cols; ++x) {
      printf(" %lld", d[x + y * h.cols]);
    }
    printf("]");
    if (y != h.rows - 1) {
      printf("\n");
    }
  }
  printf("]\n");
}

int main(int argc, char **argv) {
  int fd = open(argv[1], O_RDONLY);
  if (fd == -1) {
    perror("open");
    return EXIT_FAILURE;
  }

  struct stat stat_buf;
  CHECK_C_INT_CALL(fstat(fd, &stat_buf));

  const auto file_size = stat_buf.st_size;
#define USE_MMAP
#ifdef USE_MMAP
  void *p =
      mmap(nullptr, stat_buf.st_size, PROT_READ, MAP_FILE | MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) {
    perror("mmap");
    return EXIT_FAILURE;
  }
#else
  void *p = malloc(file_size);
  {
    auto ret = read(fd, p, file_size);
    if (ret != file_size) {
      perror("read");
      return EXIT_FAILURE;
    }
  }
#endif

  auto *data = static_cast<uint8_t *>(p);
  uint64_t working_size = 1024;
  void *working_set = aligned_alloc(sizeof(__m256i), working_size);
  RowCol m0_header = *reinterpret_cast<RowCol *>(data);
  int64_t *m0 = matrix_data(data);
  bool need_free = false;
  int counter = 0;

  std::thread threads[THREAD_COUNT];

  for (auto begin = next_block(data, data + file_size), end = data + file_size;
       ;) {
    const RowCol &m1_header = *reinterpret_cast<RowCol *>(begin);

    RowCol m2_header = {m0_header.rows, m1_header.cols};
    const auto m0_sz = m0_header.rows * m0_header.cols;
    const auto m2_sz = m2_header.rows * m2_header.cols;

    if (working_size < m2_sz + m0_sz) {
      // nearest multiple of 1024, x2
      working_size = sizeof(int64_t) * 2 * ((m2_sz + m0_sz) / 1024) * 1024;

      // freeing memory is for nerds
      if (reinterpret_cast<int64_t *>(working_set) == m0) {
        need_free = true;
      }
      working_set = aligned_alloc(sizeof(__m256i), working_size);
    }

    int64_t *m1 = matrix_data(begin);
    int64_t *m2;
    if (reinterpret_cast<int64_t *>(working_set) == m0) {
      m2 = m0 + m0_sz;
    }
    else {
      // fresh allocation.
      m2 = reinterpret_cast<int64_t *>(working_set);
    }

    if (m2_sz > THREAD_COUNT * 8 && m2_header.rows >= THREAD_COUNT) {
      for (uint32_t i = 0; i < THREAD_COUNT; ++i) {
        auto begin = i * (m2_header.rows / THREAD_COUNT);
        auto end = i == (THREAD_COUNT - 1)
                       ? m2_header.rows
                       : (i + 1) * (m2_header.rows / THREAD_COUNT);
        threads[i] = std::thread(matmul, m0_header, m1_header, m2_header, m0,
                                 m1, m2, end, begin);
      }

      for (int i = 0; i < THREAD_COUNT; ++i) {
        threads[i].join();
      }
    }
    else {
      matmul(m0_header, m1_header, m2_header, m0, m1, m2, m2_header.rows);
    }

    // matmul_noavx2(m0_header, m1_header, m2_header, m0, m1, m2);

    ++counter;

    if (need_free) {
      free(m0);
      need_free = false;
    }
    m0 = m2;
    m0_header = m2_header;

    begin = next_block(begin, end);

    if (next_block(begin, end) == nullptr) {
      break;
    }
  }

  auto f_out = fopen("output.bin", "wb");
  fwrite(&m0_header, sizeof(uint32_t), 2, f_out);
  fwrite(m0, sizeof(int64_t), m0_header.rows * m0_header.cols, f_out);
  fclose(f_out);

  // print(m0_header, m0);
}