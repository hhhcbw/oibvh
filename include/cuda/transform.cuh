#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

class Transform
{
public:
    Transform();

    ~Transform();

    /**
     * @brief         Transform glm::vec4 vector by transform matrix
     * @param[in]     vec4s               Vector4 to transform
     * @param[in]     transformMat        Transform matrix
     * @return        void
     */
    void transformVec4(std::vector<glm::vec4>& vec4s, const glm::mat4 transformMat);

private:
    glm::vec4* m_deviceVec4s;
};

/**
 * @brief         Transform glm::vec4 vector by transform matrix kernel
 * @param[in]     vec4s               Vector4 to transform
 * @param[in]     transformMat        Transform matrix
 * @param[in]     vecCount            Vector4 count
 * @return        void
 */
__global__ void transform_vec4_kernel(glm::vec4* vec4s, const glm::mat4 transformMat, const int vecCount);