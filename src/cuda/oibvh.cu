#include "cuda/oibvh.cuh"

__global__ void hello(void)
{
    printf("Hello world\n");
}

void cuda_func()
{
    hello<<<1, 1>>>();
}