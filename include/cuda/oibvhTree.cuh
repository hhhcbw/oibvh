#pragma once
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#include "utils/mesh.h"
#include "utils/utils.h"

class oibvhTree
{
public:
    oibvhTree() = delete;

    /**
     * @brief      Constructor for oibvhTree class
     * @param[in]  mesh        Mesh to build oibvh tree
     */
    oibvhTree(const std::shared_ptr<Mesh> mesh);

    /**
     * @brief   Build oibvh tree through gpu
     * @return  void
     */
    void build();

private:
    /**
     * @brief  Set up environment for building oibvh tree through gpu
     * @return void
     */
    void setup();

private:
    /**
     * @brief  Mesh to build oibvh tree
     */
    std::shared_ptr<Mesh> m_mesh;
    /**
     * @brief   AABB bvh tree of mesh
     */
    std::vector<aabb_box_t> m_aabbTree;
    /**
     * @brief   Faces of mesh with three vertices' indices
     */
    std::vector<glm::uvec3> m_faces;
    /**
     * @brief   Position of vertices in mesh
     */
    std::vector<glm::vec3> m_positions;
};