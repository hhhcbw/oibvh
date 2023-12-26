#pragma once
#include <iostream>
#include <glm/glm.hpp>

#include "utils/utils.h"
#include "cpu/simpleBVH.h"

class SimpleCollide
{
public:
    /**
     * @brief      Constructor for SimpleCollide class
     */
    SimpleCollide();

    /**
     * @brief      Deconstructor for SimpleCollide class
     */
    ~SimpleCollide();

    /**
     * @brief      Add a simple bvh to collide
     * @param[in]  simpleBvh    Simple bvh to collide
     * @return     void
     */
    void addSimpleBVH(const std::shared_ptr<SimpleBVH>& simpleBvh);

    /**
     * @brief      Detect collision between objects
     * @param[in]  printInformation         Print information or not
     * @return     void
     */
    void detect(bool printInformation = false);

    /**
     * @brief      Draw collided triangles
     * @return     void
     */
    void draw();

    /**
     * @brief         Get count of intersect triangle pairs
     * @return        Count of intersect triangle pairs
     */
    unsigned int getIntTriPairCount() const;

    /**
     * @brief      Check if giving intersect triangle pairs count are correct
     * @param[in]  intTriPairsCount          Giving count of intersect triangle pairs
     * @return     True if correct, False otherwise
     */
    bool check(const unsigned int intTriPairsCount) const;

private:
    /**
     * @brief       Convert to vertex array for rendering
     * @return      void
     */
    void convertToVertexArray();

private:
    /**
     * @brief  Vector of simple bvh trees
     */
    std::vector<std::shared_ptr<SimpleBVH>> m_simpleBvhs;
    /**
     * @brief  Intersected triangle pairs
     */
    std::vector<int_tri_pair_node_t> m_intTriPairs;
    /**
     * @brief  Vertices for rendering
     */
    std::vector<glm::vec3> m_vertices;
    /**
     *  @brief Total count of primitives of all objects
     */
    unsigned int m_primCount;
    /**
     * @brief Total count of vertices of all objects
     */
    unsigned int m_vertexCount;
    /**
     * @brief Collision detection print information times
     */
    unsigned int m_outputTimes;
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