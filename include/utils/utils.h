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
    glm::vec3 minimum;
    glm::vec3 maximum;

    bool operator==(const aabb_box& aabb) const
    {
        return minimum == aabb.minimum && maximum == aabb.maximum;
    }
} aabb_box_t;

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
 */
void make_cube(const float xSize,
               const float ySize,
               const float zSize,
               std::vector<glm::vec3>& cubeVertices,
               std::vector<unsigned int>& cubeIndices);