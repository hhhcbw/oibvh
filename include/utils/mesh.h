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
     * @brief        Copy constructor for Mesh class
     * @param[in]    other            Other mesh to copy
     */
    Mesh(const Mesh& other);

    /**
     * @brief      Deconstructor for Mesh class
     */
    ~Mesh();

    /**
     * @brief       Render the mesh with specific shader
     * @param[in]   shader               Shader to use
     * @param[in]   haveWireframe        Have wire frame or not
     * @return      void
     */
    void draw(const Shader& shader, const bool haveWireframe = false) const;

    /**
     * @brief       Rotate mesh around local axis with angle degree
     * @param[in]   axis        Rotation axis
     * @param[in]   angle       Rotation angle(degree)
     * @return      void
     */
    void rotate(const glm::vec3 axis, const float angle);

    /**
     * @brief       Rotate mesh around local x axis with angle degree
     * @param[in]   angle       Rotation angle(degree)
     * @return      void
     */
    void rotateX(const float angle = 1.0f);

    /**
     * @brief       Rotate mesh around local y axis with angle degree
     * @param[in]   angle       Rotation angle(degree)
     * @return      void
     */
    void rotateY(const float angle = 1.0f);

    /**
     * @brief       Rotate mesh around local z axis with angle degree
     * @param[in]   angle       Rotation angle(degree)
     * @return      void
     */
    void rotateZ(const float angle = 1.0f);

    /**
     * @brief       Translate mesh along with direction
     * @param[in]   translation    Translation vec3
     * @return      void
     */
    void translate(const glm::vec3 translation);

    /**
     * @brief        Transform mesh with tranform matrix
     * @param[in]    transformMat        Transform matrix
     * @return       void
     */
    void transform(const glm::mat4 transformMat);

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

    /**
     * @brief       Calculate center of mesh
     * @return      void
     */
    void setupCenter();

public:
    /**
     * @brief Count of vertices in mesh
     */
    unsigned int m_verticesCount;
    /**
     * @brief Count of faces in mesh
     */
    unsigned int m_facesCount;
    /**
     * @brief Vertices data of mesh for vertex array buffer
     */
    std::vector<Vertex> m_vertices;
    /**
     * @brief Indices of vertex data in mesh for element array buffer
     */
    std::vector<unsigned int> m_indices;
    /**
     * @brief AABB bounding box of mesh
     */
    aabb_box_t m_aabb;

private:
    /**
     * @brief Textures data of mesh
     */
    std::vector<Texture> m_textures;
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
     * @brief Center of mesh
     */
    glm::vec3 m_center;
};