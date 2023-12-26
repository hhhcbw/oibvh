#include <glm/detail/compute_common.hpp>

#include "utils/utils.h"

std::ostream& operator<<(std::ostream& os, const aabb_box_t aabb)
{
    return os << aabb.m_minimum << "X" << aabb.m_maximum;
}

std::ostream& operator<<(std::ostream& os, const glm::vec3& gvec3)
{
    return os << "(" << gvec3.x << ", " << gvec3.y << ", " << gvec3.z << ")";
}

void makeCube(const float xSize,
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

bool project6(const glm::vec3& ax,
              const glm::vec3& p1,
              const glm::vec3& p2,
              const glm::vec3& p3,
              const glm::vec3& q1,
              const glm::vec3& q2,
              const glm::vec3& q3)
{
    float P1 = glm::dot(ax, p1);
    float P2 = glm::dot(ax, p2);
    float P3 = glm::dot(ax, p3);
    float Q1 = glm::dot(ax, q1);
    float Q2 = glm::dot(ax, q2);
    float Q3 = glm::dot(ax, q3);
    float mx1 = std::fmaxf(std::fmaxf(P1, P2), P3);
    float mn1 = std::fminf(std::fminf(P1, P2), P3);
    float mx2 = std::fmaxf(std::fmaxf(Q1, Q2), Q3);
    float mn2 = std::fminf(std::fminf(Q1, Q2), Q3);

    if (mn1 > mx2)
        return false;
    if (mn2 > mx1)
        return false;
    return true;
}

bool triangleIntersect(const glm::vec3& P1,
                       const glm::vec3& P2,
                       const glm::vec3& P3,
                       const glm::vec3& Q1,
                       const glm::vec3& Q2,
                       const glm::vec3& Q3)
{
    glm::vec3 p1(0);
    glm::vec3 p2 = P2 - P1;
    glm::vec3 p3 = P3 - P1;
    glm::vec3 q1 = Q1 - P1;
    glm::vec3 q2 = Q2 - P1;
    glm::vec3 q3 = Q3 - P1;
    glm::vec3 e1 = p2 - p1;
    glm::vec3 e2 = p3 - p2;
    glm::vec3 e3 = p1 - p3;
    glm::vec3 f1 = q2 - q1;
    glm::vec3 f2 = q3 - q2;
    glm::vec3 f3 = q1 - q3;
    glm::vec3 n1 = glm::cross(e1, e2);
    glm::vec3 m1 = glm::cross(f1, f2);
    glm::vec3 g1 = glm::cross(e1, n1);
    glm::vec3 g2 = glm::cross(e2, n1);
    glm::vec3 g3 = glm::cross(e3, n1);
    glm::vec3 h1 = glm::cross(f1, m1);
    glm::vec3 h2 = glm::cross(f2, m1);
    glm::vec3 h3 = glm::cross(f3, m1);
    glm::vec3 ef11 = glm::cross(e1, f1);
    glm::vec3 ef12 = glm::cross(e1, f2);
    glm::vec3 ef13 = glm::cross(e1, f3);
    glm::vec3 ef21 = glm::cross(e2, f1);
    glm::vec3 ef22 = glm::cross(e2, f2);
    glm::vec3 ef23 = glm::cross(e2, f3);
    glm::vec3 ef31 = glm::cross(e3, f1);
    glm::vec3 ef32 = glm::cross(e3, f2);
    glm::vec3 ef33 = glm::cross(e3, f3);
    // now begin the series of tests
    if (!project6(n1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(m1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef11, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef12, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef13, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef21, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef22, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef23, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef31, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef32, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef33, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g2, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g3, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h2, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h3, p1, p2, p3, q1, q2, q3))
        return false;
    return true;
}