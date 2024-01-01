#pragma once
#include "cuda/oibvhTree.cuh"
#include "utils/mesh.h"
#include "utils/utils.h"
#include "cuda/scene.cuh"

typedef struct bvtt_node
{
    unsigned int m_aabbIndex[2];
} bvtt_node_t;

enum class DeviceType
{
    CPU = -1,
    GPU0,
    GPU1,
    GPU2,
    GPU3,
    GPU4,
    GPU5,
    GPU6,
    GPU7,
    GPU8
};

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
     * @brief         Add oibvh tree to scene
     * @param[in]     oibvhTree      Oibvh tree to add
     * @return        void
     */
    void addOibvhTree(std::shared_ptr<OibvhTree> oibvhTree);

    /**
     * @brief         Detect collision between objects
     * @param[in]     deviceId         Device id
     * @param[in]     entryLevel       Entry level of bvtt tree
     * @param[in]     expandLevels     Levels of every expansion
     * @return        void
     */
    void detectCollision(const DeviceType deviceType = DeviceType::GPU0,
                         const unsigned int entryLevel = 0,
                         const unsigned int expandLevels = 1);

    /**
     * @brief         Get count of intersect triangle pairs
     * @return        Count of intersect triangle pairs
     */
    unsigned int getIntTriPairCount() const;

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

    /**
     * @brief       Detect collision on cpu
     */
    void detectCollisionOnCPU();

    /**
     * @brief       Detect collision on gpu
     * @brief       deviceId         Id of device
     * @param[in]   entryLevel       Entry level of bvtt tree
     * @param[in]   expandLevels     Levels of every expansion
     * @return      void
     */
    void
    detectCollisionOnGPU(const unsigned int deviceId, const unsigned int entryLevel, const unsigned int expandLevels);

    /**
     * @brief       Expand initial bvtt nodes
     * @param[in]   entryLevel     Entry level of bvtt tree
     * @return      void
     */
    void expandBvttNodes(const unsigned int entryLevel);

    /**
     * @brief       Print information of giving device
     * @param[in]   deviceId          Id of device
     * @return      void
     */
    void printDeviceInfo(unsigned int deviceId);

private:
    /**
     * @brief  Vector of  oibvh trees
     */
    std::vector<std::shared_ptr<OibvhTree>> m_oibvhTrees;
    /**
     * @brief  Layout aabb offsets array
     */
    std::vector<unsigned int> m_aabbOffsets;
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
    /**
     * @brief Device count
     */
    int m_deviceCount;

    bvtt_node_t* m_deviceSrc;
    bvtt_node_t* m_deviceDst;
    tri_pair_node_t* m_deviceTriPairs;
    aabb_box_t* m_deviceAabbs;
    glm::uvec3* m_devicePrims;
    glm::vec3* m_deviceVertices;
    int_tri_pair_node_t* m_deviceIntTriPairs;
};