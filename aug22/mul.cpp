#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

#define CHECK_C_INT_CALL(call) do { int ret = call; if (ret < 0) { perror(#call); return EXIT_FAILURE; }} while (0)

struct RowCol
{
    uint32_t rows;
    uint32_t cols;
};

uint8_t* next(uint8_t* current)
{
    const RowCol& header = *reinterpret_cast<RowCol*>(current);
    return current + header.rows * header.cols * sizeof(int64_t) + sizeof(uint32_t) * 2;
}

__attribute__((noinline))
void nothing(void*) {};

int main(int argc, char** argv)
{
    int fd = open(argv[1], O_RDONLY);
    if (fd == -1)
    {
        perror("open");
        return EXIT_FAILURE;
    }

    struct stat stat_buf;
    CHECK_C_INT_CALL(fstat(fd, &stat_buf));

    const auto file_size = stat_buf.st_size;
// #define USE_MMAP
#ifdef USE_MMAP
    void* p = mmap(nullptr, stat_buf.st_size, PROT_READ, MAP_FILE | MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED)
    {
        perror("mmap");
        return EXIT_FAILURE;
    }
#else
    void* p = malloc(file_size);
    {
        auto ret = read(fd, p, file_size);
        if (ret != file_size)
        {
            perror("read");
            return EXIT_FAILURE;
        }
    }
#endif

    auto* data = static_cast<uint8_t*>(p);
    for (auto begin = data; begin != data + file_size;)
    {
        const RowCol& h = *reinterpret_cast<RowCol*>(begin);
        
        nothing((void*)&h);
        begin = next(begin);
    }
}