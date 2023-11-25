/*********************************************************************
 * @file       mesh.h
 * @brief      Header file for Mesh class
 * @details
 * @author     hhhcbw
 * @date       2023-11-20
 *********************************************************************/
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <string>
#include <vector>

#include "shader.h"
#include "utils.h"

#define MAX_BONE_INFLUENCE 4

struct Vertex
{
    /**
     * @brief Position coordinate of vertex
     */
    glm::vec3 m_position;
    /**
     * @brief Normal vector of vertex
     */
    glm::vec3 m_normal;
    /**
     * @brief Texture coordinate of vertex
     */
    glm::vec2 m_texCoords;
    /**
     * @brief Tangent vector of vertex
     */
    glm::vec3 m_tangent;
    /**
     * @brief Bitangent vector of vertex
     */
    glm::vec3 m_bitangent;
    /**
     * @brief Bone indexes which will influence this vertex
     */
    int m_boneIds[MAX_BONE_INFLUENCE];
    /**
     * @brief Weights from each bone
     */
    float m_weights[MAX_BONE_INFLUENCE];
};

struct Texture
{
    /**
     * @brief Id of texture
     */
    unsigned int m_id;
    /**
     * @brief Type of texture
     */
    std::string m_type;
    /**
     * @brief File path of texture
     */
    std::string m_path;
};

class Mesh
{
public:
    Mesh() = delete;

    /**
     * @brief       Constructor for Mesh class
     * @param[in]   vertices    Vertices data of mesh for vertex array buffer
     * @param[in]   indices     Indices of vertex data in mesh for element array buffer
     * @param[in]   textures    Textures data of mesh
     */
    Mesh(const std::vector<Vertex>& vertices,
         const std::vector<unsigned int>& indices,
         const std::vector<Texture>& textures = std::vector<Texture>());

    /**
     * @brief      Deconstructor for Mesh class
     */
    ~Mesh();

    /**
     * @brief       Render the mesh with specific shader
     * @param[in]   shader     Shader to use
     * @return      void
     */
    void draw(const Shader& shader) const;

    /**
     * @brief       Get aabb bounding box of mesh
     * @return      Bounding box of mesh
     */
    aabb_box_t getAABB() const;

private:
    /**
     * @brief       Set aabb bounding box of mesh
     * @return      void
     */
    void setupAABB();

    /**
     * @brief     Initializes all the buffer objects / arrays
     * @return    void
     */
    void setupMesh();

public:
    /**
     * @brief Count of vertices in mesh
     */
    unsigned int m_verticesCount;
    /**
     * @brief Count of faces in mesh
     */
    unsigned int m_facesCount;

private:
    /**
     * @brief Vertices data of mesh for vertex array buffer
     */
    std::vector<Vertex> m_vertices;
    /**
     * @brief Indices of vertex data in mesh for element array buffer
     */
    std::vector<unsigned int> m_indices;
    /**
     * @brief Textures data of mesh
     */
    std::vector<Texture> m_textures;
    /**
     * @brief AABB bounding box of mesh
     */
    aabb_box_t m_aabb;
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
};