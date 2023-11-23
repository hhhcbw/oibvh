#include <glad/glad.h> // holds all OpenGL type declarations

#include <iostream>

#include "mesh.h"

Mesh::Mesh(const std::vector<Vertex>& vertices,
           const std::vector<unsigned int>& indices,
           const std::vector<Texture>& textures)
    : m_vertices(vertices)
    , m_indices(indices)
    , m_textures(textures)
    , m_verticesCount(vertices.size())
    , m_facesCount(indices.size() / 3)
{
    // calculate bounding box of mesh
    setupAABB();
    // now that we have all the required data, set the vertex buffers and its attribute pointers.
    setupMesh();
}

Mesh::~Mesh()
{
    glDeleteVertexArrays(1, &m_vertexArrayObj);
    glDeleteBuffers(1, &m_vertexBufferObj);
    glDeleteBuffers(1, &m_elementBufferObj);
}

void Mesh::draw(const Shader& shader) const
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
    glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(m_indices.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // always good practice to set everything back to defaults once configured.
    glActiveTexture(GL_TEXTURE0);
}

aabb_box_t Mesh::getAABB() const
{
    return m_aabb;
}

void Mesh::setupAABB()
{
    m_aabb.minimum = glm::vec3(1e10);
    m_aabb.maximum = glm::vec3(-1e10);
    for (const auto& vertex : m_vertices)
    {
        m_aabb.maximum.x = std::fmax(vertex.m_position.x, m_aabb.maximum.x);
        m_aabb.maximum.y = std::fmax(vertex.m_position.y, m_aabb.maximum.y);
        m_aabb.maximum.z = std::fmax(vertex.m_position.z, m_aabb.maximum.z);

        m_aabb.minimum.x = std::fmin(vertex.m_position.x, m_aabb.minimum.x);
        m_aabb.minimum.y = std::fmin(vertex.m_position.y, m_aabb.minimum.y);
        m_aabb.minimum.z = std::fmin(vertex.m_position.z, m_aabb.minimum.z);
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
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(Vertex), &m_vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elementBufferObj);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(unsigned int), &m_indices[0], GL_STATIC_DRAW);

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