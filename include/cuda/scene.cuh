#pragma once
#include <utility>
#include "cuda/oibvhTree.cuh"
#include "utils/mesh.h"
#include "cuda/scene.cuh"

typedef struct bvtt_node
{
    unsigned int m_bvhIndex[2];
} bvtt_node_t;

typedef struct tri_pair_node
{
    unsigned int m_triIndex[2];
} tri_pair_node_t;

typedef struct int_tri_pair_node
{
    unsigned int m_meshIndex[2];
    unsigned int m_triIndex[2];
} int_tri_pair_node_t;

class Scene
{
public:
    /**
     * @brief         Constructor for Scene
     */
    Scene();

    /**
     * @brief         Deconstructor for Scene
     */
    ~Scene();

    /**
     * @brief         Add object to scene
     * @param[in]     object        Object to add
     * @return        void
     */
    void addObject(std::pair<std::shared_ptr<Mesh>, std::shared_ptr<OibvhTree>> object);

    /**
     * @brief         Detect collision between objects
     * @return        void
     */
    void detectCollision();

    /**
     * @brief         Draw collided triangles
     * @return        void
     */
    void draw();

private:
    /**
     * @brief       Convert to vertex array for rendering
     * @return      void
     */
    void convertToVertexArray();

private:
    /**
     * @brief  Vector of pairs of mesh and its corresponding oibvh tree
     */
    std::vector<std::pair<std::shared_ptr<Mesh>, std::shared_ptr<OibvhTree>>> m_objects;
    /**
     * @brief  Layout bvh offsets array
     */
    std::vector<unsigned int> m_bvhOffsets;
    /**
     * @brief  Layout primitive offsets array
     */
    std::vector<unsigned int> m_primOffsets;
    /**
     * @brief  Layout vertices offsets array
     */
    std::vector<unsigned int> m_vertexOffsets;
    /**
     * @brief  Layout primitive count array
     */
    std::vector<unsigned int> m_primCounts;
    /**
     * @brief  Initial bvtt nodes array
     */
    std::vector<bvtt_node_t> m_bvtts;
    /**
     * @brief  Intersected triangle pairs
     */
    std::vector<int_tri_pair_node_t> m_intTriPairs;
    /**
     * @brief  Vertices for rendering
     */
    std::vector<glm::vec3> m_vertices;
    /**
     * @brief  Total count of aabb boxes of all objects
     */
    unsigned int m_aabbCount;
    /**
     *  @brief Total count of primitives of all objects
     */
    unsigned int m_primCount;
    /**
     * @brief Total count of vertices of all objects
     */
    unsigned int m_vertexCount;
    /**
     * @brief Collision detection times
     */
    unsigned int m_detectTimes;
    /**
     * @brief Count of intersected triangle pairs
     */
    unsigned int m_intTriPairCount;
    /**
     * @brief Vertex arrays object id
     */
    unsigned int m_vertexArrayObj;
    /**
     * @brief Vertex buffer object id
     */
    unsigned int m_vertexBufferObj;
    /**
     * @brief Have convert to vertices array done or not
     */
    bool m_convertDone;
};