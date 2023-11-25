/*********************************************************************
 * @file       utils.h
 * @brief      Untility type and function
 * @details
 * @author     hhhcbw
 * @date       2023-11-23
 *********************************************************************/
#pragma once

#include <glm/glm.hpp>

typedef struct
{
    glm::vec3 minimum;
    glm::vec3 maximum;
} aabb_box_t;