#pragma once

#include <vector>

// public owl node-graph API
#include <owl/owl.h>
// our device-side data structures
#include "cpu/deviceCode.h"
#include "utils/mesh.h"

#define LOG(message)                                            \
    std::cout << OWL_TERMINAL_BLUE;                             \
    std::cout << "#owl.sample(main): " << message << std::endl; \
    std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
    std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
    std::cout << "#owl.sample(main): " << message << std::endl; \
    std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

class OptixCollide
{
public:
    OptixCollide(std::vector<std::shared_ptr<Mesh>>& meshes);
    ~OptixCollide();
    void init();
    void detect();
    void draw();
    void refit();

private:
    void convertToVertexArray();

private:
    /**
     * @brief Vertex arrays object id
     */
    unsigned int m_vertexArrayObj;
    /**
     * @brief Vertex buffer object id
     */
    unsigned int m_vertexBufferObj;
    std::vector<glm::vec3> m_drawVertices;
    bool m_convertDone;

    OWLContext m_context;
    std::vector<std::shared_ptr<Mesh>> m_meshes;
    OWLRayGen m_rayGen;
    OWLBuffer m_vertexBuffer;
    OWLBuffer m_indexBuffer;
    OWLBuffer m_auxVertexBuffer;
    OWLBuffer m_auxIndexBuffer;
    OWLBuffer m_indexPairsBuffer;
    OWLBuffer m_countBuffer;
    OWLGeom m_trianglesGeom;
    OWLGeom m_auxTrianglesGeom;
    OWLGroup m_trianglesGroup;
    std::vector<owl::vec2ui> m_indexPairs;
    std::vector<unsigned int> m_faceCounts;
    std::vector<owl::vec3f> m_vertices;
    std::vector<owl::vec3ui> m_indices;
    std::vector<owl::vec3f> m_auxVertices;
    std::vector<owl::vec3ui> m_auxIndices;

    unsigned int m_detectTimes;
};