#include "utils/utils.h"

std::ostream& operator<<(std::ostream& os, const glm::vec3& gvec3)
{
    return os << "(" << gvec3.x << ", " << gvec3.y << ", " << gvec3.z << ")";
}

void make_cube(const float xSize,
               const float ySize,
               const float zSize,
               std::vector<glm::vec3>& cubeVertices,
               std::vector<unsigned int>& cubeIndices)
{
    // front quad
    cubeVertices.push_back(glm::vec3(-xSize, -ySize, zSize)); // 0
    cubeVertices.push_back(glm::vec3(xSize, -ySize, zSize));  // 1
    cubeVertices.push_back(glm::vec3(xSize, ySize, zSize));   // 2
    cubeVertices.push_back(glm::vec3(-xSize, ySize, zSize));  // 3

    // back quad
    cubeVertices.push_back(glm::vec3(-xSize, -ySize, -zSize)); // 4
    cubeVertices.push_back(glm::vec3(xSize, -ySize, -zSize));  // 5
    cubeVertices.push_back(glm::vec3(xSize, ySize, -zSize));   // 6
    cubeVertices.push_back(glm::vec3(-xSize, ySize, -zSize));  // 7

    // front
    cubeIndices.push_back(0U);
    cubeIndices.push_back(1U);
    /**/
    cubeIndices.push_back(1U);
    cubeIndices.push_back(2U);
    /**/
    cubeIndices.push_back(2U);
    cubeIndices.push_back(3U);
    /**/
    cubeIndices.push_back(3U);
    cubeIndices.push_back(0U);
    // back
    cubeIndices.push_back(4U);
    cubeIndices.push_back(5U);
    /**/
    cubeIndices.push_back(5U);
    cubeIndices.push_back(6U);
    /**/
    cubeIndices.push_back(6U);
    cubeIndices.push_back(7U);
    /**/
    cubeIndices.push_back(7U);
    cubeIndices.push_back(4U);
    // side lines
    cubeIndices.push_back(1U);
    cubeIndices.push_back(5U);
    /**/
    cubeIndices.push_back(0U);
    cubeIndices.push_back(4U);
    /**/
    cubeIndices.push_back(3U);
    cubeIndices.push_back(7U);
    /**/
    cubeIndices.push_back(2U);
    cubeIndices.push_back(6U);
}