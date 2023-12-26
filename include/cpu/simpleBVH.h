#pragma once
#include <iostream>

#include <glm/glm.hpp>

#include "utils/mesh.h"
#include "utils/utils.h"

typedef struct simple_bvh_node
{
    aabb_box_t m_aabb;
    std::shared_ptr<simple_bvh_node> m_left;
    std::shared_ptr<simple_bvh_node> m_right;
    int m_triId; // -1 if not a leaf node
    simple_bvh_node() : m_aabb(), m_left(nullptr), m_right(nullptr), m_triId(-1)
    {
    }
} simple_bvh_node_t;

class SimpleBVH
{
public:
    SimpleBVH() = delete;

    /**
     * @brief      Constructor for SimpleBVH class
     * @param[in]  mesh        Mesh to build simple bvh
     */
    SimpleBVH(const std::shared_ptr<Mesh> mesh);

    /**
     * @brief      Build simple bvh
     * @return     void
     */
    void build();

    /**
     * @brief     Refit simple bvh
     * @return    void
     */
    void refit();

    /**
     * @brief     Log simple bvh
     * @param[in] path       Path to log file
     * @return    void
     */
    void log(const std::string& path = std::string("C://Code//oibvh//logs//simple_bvh_log.txt")) const;

    /**
     * @brief   Mesh updates and simple bvh is unrefit
     * @return  void
     */
    void unRefit();

private:
    /**
     * @brief      Build simple bvh recursively
     * @param[in]  leftPrim     Left primitive index
     * @param[in]  rightPrim    Right primitive index
     * @param[in]  depth        Depth of current node
     * @return     Root node of simple bvh tree
     */
    std::shared_ptr<simple_bvh_node_t> recursiveBuild(int leftPrim, int rightPrim, int depth = 0);

    /**
     * @brief      Build simple bvh recursively
     * @param[in]  node         Simple bvh node
     * @return     void
     */
    void recursiveRefit(const std::shared_ptr<simple_bvh_node_t>& node) const;

private:
    /**
     * @brief  Mesh to build oibvh tree
     */
    std::shared_ptr<Mesh> m_mesh;
    /**
     * @brief   Faces of mesh with three vertices' indices
     */
    std::vector<glm::uvec3> m_faces;
    /**
     * @brief   Position of vertices in mesh
     */
    std::vector<glm::vec3> m_positions;
    /**
     * @brief   Have build bvh done or not
     */
    bool m_buildDone;
    /**
     * @brief   Have refit bvh done or not
     */
    bool m_refitDone;
    /**
     * @brief   Root node of bvh tree
     */
    std::shared_ptr<simple_bvh_node_t> m_root;
    /**
     * @brief   Count of nodes in simple bvh tree
     */
    unsigned int m_nodeCount;
    /**
     * @brief   Depth of simple bvh tree(from 0)
     */
    unsigned int m_depth;

    friend class SimpleCollide;
};