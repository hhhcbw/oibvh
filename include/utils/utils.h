/*********************************************************************
 * @file       utils.h
 * @brief      Untility type and function
 * @details
 * @author     hhhcbw
 * @date       2023-11-23
 *********************************************************************/
#pragma once

#include <iostream>
#include <vector>

#include <glm/glm.hpp>

typedef struct aabb_box
{
    glm::vec3 m_minimum{std::numeric_limits<float>::max()};
    glm::vec3 m_maximum{-std::numeric_limits<float>::max()};

    void init(const aabb_box& other)
    {
        m_minimum = other.m_minimum;
        m_maximum = other.m_maximum;
    }

    void init(const glm::vec3& minimum = glm::vec3(std::numeric_limits<float>::max()),
              const glm::vec3& maximum = glm::vec3(-std::numeric_limits<float>::max()))
    {
        m_minimum = minimum;
        m_maximum = maximum;
    }

    void merge(const aabb_box& other)
    {
        m_minimum = glm::min(m_minimum, other.m_minimum);
        m_maximum = glm::max(m_maximum, other.m_maximum);
    }

    bool overlap(const aabb_box& other)
    {
        return (m_minimum.x <= other.m_maximum.x && m_maximum.x >= other.m_minimum.x) &&
            (m_minimum.y <= other.m_maximum.y && m_maximum.y >= other.m_minimum.y) &&
            (m_minimum.z <= other.m_maximum.z && m_maximum.z >= other.m_minimum.z);
    }

    bool operator==(const aabb_box& other) const
    {
        return m_minimum == other.m_minimum && m_maximum == other.m_maximum;
    }
} aabb_box_t;

typedef struct tri_pair_node
{
    unsigned int m_triIndex[2];
} tri_pair_node_t;

typedef struct int_tri_pair_node
{
    unsigned int m_bvhIndex[2];
    unsigned int m_triIndex[2];
} int_tri_pair_node_t;

/**
 * @brief       Overload operator for aabb_box
 * @param[in]   os       Ouput stream
 * @param[in]   aabb     aab_box to output
 * @return      Output stream
 */
std::ostream& operator<<(std::ostream& os, const aabb_box_t aabb);

/**
 * @brief 	  Overload operator<< for glm::vec3
 * @param[in] os       Output stream
 * @param[in] gvec3    glm::vec3 to output
 * @return    Output stream
 */
std::ostream& operator<<(std::ostream& os, const glm::vec3& gvec3);

/**
 * @brief      Make cube vertices array and indices for aabb bounding box
 * @param[in]  xSize              Half of x size of aabb bounding box
 * @param[in]  ySize              Half of y size of aabb bounding box
 * @param[in]  zSize              Half of z size of aabb bounding box
 * @param[in]  cubeVertices       Vertices array of cube
 * @param[in]  cubeIndices        Indices array of cube
 * @return     void
 */
void makeCube(const float xSize,
              const float ySize,
              const float zSize,
              std::vector<glm::vec3>& cubeVertices,
              std::vector<unsigned int>& cubeIndices);

/**
 * @brief     Check if two triangles intersect
 * @param[in] P1       First vertex of triangle 1
 * @param[in] P2       Second vertex of triangle 1
 * @param[in] P3       Third vertex of triangle 1
 * @param[in] Q1       First vertex of triangle 2
 * @param[in] Q2       Second vertex of triangle 2
 * @param[in] Q3       Third vertex of triangle 2
 * @return    True is intersect, otherwise false
 */
bool triangleIntersect(const glm::vec3& P1,
                       const glm::vec3& P2,
                       const glm::vec3& P3,
                       const glm::vec3& Q1,
                       const glm::vec3& Q2,
                       const glm::vec3& Q3);