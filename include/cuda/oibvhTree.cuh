#pragma once
#include <iostream>
#include <vector>

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

    /**
     * @brief         Draw mesh and bounding box(drawBox = true)
     * @param[in]     shader        shader to use on box
     * @return        void
     */
    void draw(const Shader& shader);

private:
    /**
     * @brief  Set up environment for building oibvh tree through gpu
     * @return void
     */
    void setup();

    /**
     * @brief  Convert bvh data to vertex array for rendering
     * @return void
     */
    void convertToVertexArray();

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
    /**
     * @brief   Have converted bvh data to vertex array for rendering or not
     */
    bool m_convertDone;
    /**
     * @brief   Have build bvh done or not
     */
    bool m_buildDone;
    /**
     * @brief Vertex arrays object id
     */
    unsigned int m_vertexArrayObj;
    /**
     * @brief Vertex buffer object id
     */
    unsigned int m_vertexBufferObj;
    /**
     * @brief Element buffer object id
     */
    unsigned int m_elementBufferObj;
    /**
     * @brief Vertex array for bvh
     */
    std::vector<glm::vec3> m_vertices;
    /**
     * @brief Indices of vertex array for bvh
     */
    std::vector<unsigned int> m_indices;
};