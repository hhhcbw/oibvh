#include <iostream>

#include <device_launch_parameters.h>

#include "cuda/transform.cuh"
#include "cuda/utils.cuh"

Transform::Transform()
{
    deviceMalloc(&m_deviceVec4s, 10000000);
}

Transform::~Transform()
{
	cudaFree(m_deviceVec4s);
}

void Transform::transformVec4(std::vector<glm::vec4>& vec4s, const glm::mat4 transformMat)
{
    glm::vec4* d_vec4s = m_deviceVec4s;
    deviceMemcpy(d_vec4s, vec4s.data(), vec4s.size());

    float elapsed_ms = 0.0f;

    elapsed_ms = kernelLaunch([&]() {
        dim3 blockSize = dim3(256);
        int bx = (vec4s.size() + blockSize.x - 1) / blockSize.x;
        dim3 gridSize = dim3(bx);
        transform_vec4_kernel<<<gridSize, blockSize>>>(d_vec4s, transformMat, vec4s.size());
    });

    hostMemcpy(vec4s.data(), d_vec4s, vec4s.size());
}

__global__ void transform_vec4_kernel(glm::vec4* vec4s, const glm::mat4 transformMat, const int vecCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vecCount)
        return;
    vec4s[idx] = transformMat * vec4s[idx];
}