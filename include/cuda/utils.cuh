#pragma once
#include <cuda_runtime.h>
#include <cuda/std/bit>
#include <functional>

template <typename T>
inline void deviceMalloc(T** ptr, size_t size)
{
    cudaMalloc(ptr, size * sizeof(T));
}

template <typename T>
inline void hostMalloc(T** ptr, size_t size)
{
    *ptr = new T[size];
}

template <typename T>
inline void deviceMemcpy(T* dst, T* src, size_t size)
{
    cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
inline void hostMemcpy(T* dst, T* src, size_t size)
{
    cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline void deviceMemset(T* ptr, int value, size_t size)
{
    cudaMemset(ptr, value, size * sizeof(T));
}

inline float kernelLaunch(std::function<void()> func, bool measure_time = true)
{
    float ms = 0.0f;
    if (measure_time)
    {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        func();
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    else
    {
        func();
    }

    return ms;
}

/**
 * @brief        Next power of two of x
 * @param[in]    x    The value to round up
 * @return       The next power of two
 */
__device__ __host__ inline unsigned int next_power_of_two(unsigned int x)
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

/**
 * @brief        Round down log2(x)
 * @param[in]    x    The value to take the log2 of
 * @return       The floor of log2(x)
 */
__device__ __host__ inline unsigned int ilog2(unsigned int x)
{
    return sizeof(unsigned int) * CHAR_BIT - cuda::std::__countl_zero(x) - 1;
}