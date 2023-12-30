#include <glad/glad.h> // holds all OpenGL type declarations

#include <iostream>

#include "utils/mesh.h"

Mesh::Mesh(const std::vector<Vertex>& vertices,
           const std::vector<unsigned int>& indices,
           const std::vector<Texture>& textures)
    : m_vertices(vertices)
    , m_indices(indices)
    , m_textures(textures)
    , m_verticesCount(vertices.size())
    , m_facesCount(indices.size() / 3)
{
    m_newVertices.resize(m_verticesCount);
    // calculate bounding box of mesh
    setupAABB();
    // now that we have all the required data, set the vertex buffers and its attribute pointers.
    setupMesh();
    // calculate center of mesh
    setupCenter();
}

Mesh::Mesh(const Mesh& other)
    : m_vertices(other.m_vertices)
    , m_indices(other.m_indices)
    , m_textures(other.m_textures)
    , m_verticesCount(other.m_verticesCount)
    , m_facesCount(other.m_facesCount)
    , m_aabb(other.m_aabb)
    , m_center(other.m_center)
{
    m_newVertices.resize(m_verticesCount);
    setupMesh();
}

Mesh::~Mesh()
{
    glDeleteVertexArrays(1, &m_vertexArrayObj);
    glDeleteBuffers(1, &m_vertexBufferObj);
    glDeleteBuffers(1, &m_elementBufferObj);
}

void Mesh::draw(const Shader& shader, const bool haveWireframe) const
{
    // bind appropriate textures
    unsigned int diffuseNr = 1;
    unsigned int specularNr = 1;
    unsigned int normalNr = 1;
    unsigned int heightNr = 1;
    for (unsigned int i = 0; i < m_textures.size(); i++)
    {
        glActiveTexture(GL_TEXTURE0 + i); // active proper texture unit before binding
        // retrieve texture number (the N in diffuse_textureN)
        std::string number;
        std::string name = m_textures[i].m_type;
        if (name == "texture_diffuse")
            number = std::to_string(diffuseNr++);
        else if (name == "texture_specular")
            number = std::to_string(specularNr++); // transfer unsigned int to string
        else if (name == "texture_normal")
            number = std::to_string(normalNr++); // transfer unsigned int to string
        else if (name == "texture_height")
            number = std::to_string(heightNr++); // transfer unsigned int to string

        // now set the sampler to the correct texture unit
        shader.setInt(name + number, i);
        // and finally bind the texture
        glBindTexture(GL_TEXTURE_2D, m_textures[i].m_id);
    }

    // draw mesh
    glBindVertexArray(m_vertexArrayObj);
    shader.setBool("wireframe", false);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(m_indices.size()), GL_UNSIGNED_INT, 0);
    if (haveWireframe)
    {
        shader.setBool("wireframe", true);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(m_indices.size()), GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    glBindVertexArray(0);

    // always good practice to set everything back to defaults once configured.
    glActiveTexture(GL_TEXTURE0);
}

void Mesh::setupAABB()
{
    for (const auto& vertex : m_vertices)
    {
        m_aabb.m_maximum = glm::max(vertex.m_position, m_aabb.m_maximum);
        m_aabb.m_minimum = glm::min(vertex.m_position, m_aabb.m_minimum);
    }
}

void Mesh::setupMesh()
{
    // create buffers/arrays
    glGenVertexArrays(1, &m_vertexArrayObj);
    glGenBuffers(1, &m_vertexBufferObj);
    glGenBuffers(1, &m_elementBufferObj);

    glBindVertexArray(m_vertexArrayObj);
    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);
    // A great thing about structs is that their memory layout is sequential for all its items.
    // The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2
    // array which again translates to 3/2 floats which translates to a byte array.
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(Vertex), m_vertices.data(), GL_STREAM_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elementBufferObj);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(unsigned int), m_indices.data(), GL_STATIC_DRAW);

    // set the vertex attribute pointers
    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    //// vertex normals
    // glEnableVertexAttribArray(1);
    // glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, m_normal));
    //// vertex texture coords
    // glEnableVertexAttribArray(2);
    // glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, m_texCoords));
    //// vertex tangent
    // glEnableVertexAttribArray(3);
    // glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, m_tangent));
    //// vertex bitangent
    // glEnableVertexAttribArray(4);
    // glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, m_bitangent));
    //// ids
    // glEnableVertexAttribArray(5);
    // glVertexAttribIPointer(5, MAX_BONE_INFLUENCE, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, m_boneIds));
    //// weights
    // glEnableVertexAttribArray(6);
    // glVertexAttribPointer(
    //     6, MAX_BONE_INFLUENCE, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, m_weights));

    glBindVertexArray(0);
}

void Mesh::setupCenter()
{
    m_center = glm::vec3(0.0f);
    for (const auto& vertex : m_vertices)
    {
        m_center += vertex.m_position;
    }
    m_center /= (float)m_verticesCount;
}

void Mesh::rotateX(const float angle)
{
    rotate(glm::vec3(1.0f, 0.0f, 0.0f), angle);
}

void Mesh::rotateY(const float angle)
{
    rotate(glm::vec3(0.0f, 1.0f, 0.0f), angle);
}

void Mesh::rotateZ(const float angle)
{
    rotate(glm::vec3(0.0f, 0.0f, 1.0f), angle);
}

void Mesh::rotate(const glm::vec3 axis, const float angle)
{
    glm::mat4 transformMat = glm::identity<glm::mat4>();
    transformMat = glm::translate(transformMat, m_center);
    transformMat = glm::rotate(transformMat, glm::radians(angle), axis);
    transformMat = glm::translate(transformMat, -m_center);

    transform(transformMat);
}

void Mesh::translate(const glm::vec3 translation)
{
    glm::mat4 transformMat = glm::identity<glm::mat4>();
    transformMat = glm::translate(transformMat, translation);
    transform(transformMat);
}

void Mesh::transform(const glm::mat4 transformMat)
{
    m_center = glm::vec3(transformMat * glm::vec4(m_center, 1.0f));
    /*std::vector<glm::vec4> newVertices(m_vertices.size());*/
    // std::vector<glm::vec4> newNormals(m_vertices.size());
    for (int i = 0; i < m_verticesCount; i++)
    {
        m_newVertices[i] = glm::vec4(m_vertices[i].m_position, 1.0f);
        // newNormals[i] = glm::vec4(m_vertices[i].m_normal, 0.0f);
    }

    m_transform.transformVec4(m_newVertices, transformMat);
    // transformVec4(newNormals, transformMat);

    for (int i = 0; i < m_verticesCount; i++)
    {
        m_vertices[i].m_position = glm::vec3(m_newVertices[i]);
        // m_vertices[i].m_normal = glm::vec3(newNormals[i]);
    }

    glBindVertexArray(m_vertexArrayObj);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);

    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(Vertex), m_vertices.data(), GL_STREAM_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}