/*********************************************************************
 * @file       utils.h
 * @brief      Untility type and function
 * @details
 * @author     hhhcbw
 * @date       2023-11-23
 *********************************************************************/
#pragma once

#include <iostream>

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