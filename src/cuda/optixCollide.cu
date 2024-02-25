#include <iostream>
#include <fstream>

#include "cuda/optixCollide.cuh"
#include "cuda/utils.cuh"
#include "utils/utils.h"

#define LOG_TIMES 100

void genAuxTriangles(const std::vector<owl::vec3f>& vertices,
                     const std::vector<owl::vec3ui>& indices,
                     std::vector<owl::vec3f>& auxVertices,
                     std::vector<owl::vec3ui>& auxIndices)
{
    auxVertices.resize(indices.size() * 9);
    auxIndices.resize(indices.size() * 3);
    for (unsigned int i = 0; i < indices.size(); i++)
    {
        owl::vec3f v[3];
        v[0] = vertices[indices[i].x];
        v[1] = vertices[indices[i].y];
        v[2] = vertices[indices[i].z];
        owl::vec3f dir1 = owl::normalize(v[1] - v[0]);
        owl::vec3f dir2 = owl::normalize(v[2] - v[0]);
        owl::vec3f norm = owl::normalize(cross(dir1, dir2));
        for (unsigned int j = 0; j < 3; j++)
        {
            auxVertices[i * 9 + j * 3 + 0] = v[j] + 0.1f * norm;
            auxVertices[i * 9 + j * 3 + 1] = v[j] - 0.1f * norm;
            auxVertices[i * 9 + j * 3 + 2] = v[(j + 1) % 3];
            auxIndices[i * 3 + j] = owl::vec3ui{i * 9 + j * 3 + 0, i * 9 + j * 3 + 1, i * 9 + j * 3 + 2};
        }
    }
}

void dumplicationRemove(std::vector<owl::vec2ui>& indexPairs)
{
    for (auto& indexPair : indexPairs)
    {
        if (indexPair.x > indexPair.y)
        {
            std::swap(indexPair.x, indexPair.y);
        }
    }
    std::sort(indexPairs.begin(), indexPairs.end());
    indexPairs.erase(std::unique(indexPairs.begin(), indexPairs.end()), indexPairs.end());
}

int getMeshID(int faceIndex, std::vector<unsigned int>& faceCounts)
{
    int l = 0;
    int r = faceCounts.size() - 1;
    int ans = 0;
    while (l <= r)
    {
        int mid = (l + r) / 2;
        if (faceCounts[mid] == faceIndex)
        {
            return mid;
        }
        else if (faceCounts[mid] < faceIndex)
        {
            ans = mid;
            l = mid + 1;
        }
        else
            r = mid - 1;
    }
    return ans;
}

OptixCollide::OptixCollide(std::vector<std::shared_ptr<Mesh>>& meshes) : m_meshes(meshes), m_detectTimes(0)
{
    glGenVertexArrays(1U, &m_vertexArrayObj);
    glGenBuffers(1U, &m_vertexBufferObj);

    glBindVertexArray(m_vertexArrayObj);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * m_drawVertices.size(), m_drawVertices.data(), GL_STREAM_DRAW);

    glEnableVertexAttribArray(0U);
    glVertexAttribPointer(0U, 3U, GL_FLOAT, GL_FALSE, 0, (void*)0); // position

    glBindVertexArray(0U);
}

OptixCollide::~OptixCollide()
{
    glDeleteVertexArrays(1, &m_vertexArrayObj);
    glDeleteBuffers(1, &m_vertexBufferObj);

    LOG("destroying devicegroup ...");
    owlContextDestroy(m_context);

    LOG_OK("seems all went OK; app is done, this should be the last output ...");
}

void OptixCollide::draw()
{
    if (!m_convertDone)
    {
        convertToVertexArray();
        m_convertDone = true;
    }

    glBindVertexArray(m_vertexArrayObj);
    glDrawArrays(GL_TRIANGLES, 0, m_drawVertices.size());
    glBindVertexArray(0U);
}

void OptixCollide::convertToVertexArray()
{
    m_drawVertices.resize(m_indexPairs.size() * 6);
    for (int i = 0; i < m_indexPairs.size(); i++)
    {
        const auto intTriPair = m_indexPairs[i];
        const auto meshID1 = getMeshID(intTriPair.x, m_faceCounts);
        const auto meshID2 = getMeshID(intTriPair.y, m_faceCounts);
        assert(meshID1 != meshID2);
        const auto mesh1 = m_meshes[meshID1];
        const auto mesh2 = m_meshes[meshID2];
        const auto index1a = mesh1->m_indices[(intTriPair.x - m_faceCounts[meshID1]) * 3 + 0];
        const auto index1b = mesh1->m_indices[(intTriPair.x - m_faceCounts[meshID1]) * 3 + 1];
        const auto index1c = mesh1->m_indices[(intTriPair.x - m_faceCounts[meshID1]) * 3 + 2];
        const auto index2a = mesh2->m_indices[(intTriPair.y - m_faceCounts[meshID2]) * 3 + 0];
        const auto index2b = mesh2->m_indices[(intTriPair.y - m_faceCounts[meshID2]) * 3 + 1];
        const auto index2c = mesh2->m_indices[(intTriPair.y - m_faceCounts[meshID2]) * 3 + 2];
        m_drawVertices[i * 6] = mesh1->m_vertices[index1a].m_position;
        m_drawVertices[i * 6 + 1] = mesh1->m_vertices[index1b].m_position;
        m_drawVertices[i * 6 + 2] = mesh1->m_vertices[index1c].m_position;
        m_drawVertices[i * 6 + 3] = mesh2->m_vertices[index2a].m_position;
        m_drawVertices[i * 6 + 4] = mesh2->m_vertices[index2b].m_position;
        m_drawVertices[i * 6 + 5] = mesh2->m_vertices[index2c].m_position;
    }

    // for (int i = 0; i < m_indexPairs.size(); i++)
    //{
    //     if (!triangleIntersect(m_drawVertices[i * 6], m_drawVertices[i * 6 + 1], m_drawVertices[i * 6 + 2],
    //     m_drawVertices[i * 6 + 3], m_drawVertices[i * 6 + 4], m_drawVertices[i * 6 + 5]))
    //     {
    //         printf("error\n");
    //         break;
    //     }
    // }

    glBindVertexArray(m_vertexArrayObj);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertexBufferObj);
    glBufferData(GL_ARRAY_BUFFER, m_drawVertices.size() * sizeof(glm::vec3), m_drawVertices.data(), GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    m_convertDone = true;
}

void OptixCollide::init()
{
    // prepare vertex and index data and generate auxilury triangles
    unsigned int indexOffset = 0;
    unsigned int faceCount = 0;
    for (auto& mesh : m_meshes)
    {
        for (auto& vertex : mesh->m_vertices)
        {
            m_vertices.push_back({vertex.m_position.x, vertex.m_position.y, vertex.m_position.z});
        }
        for (int i = 0; i < mesh->m_facesCount; i++)
        {
            m_indices.push_back({
                mesh->m_indices[i * 3 + 0] + indexOffset,
                mesh->m_indices[i * 3 + 1] + indexOffset,
                mesh->m_indices[i * 3 + 2] + indexOffset,
            });
        }
        m_faceCounts.push_back(faceCount);
        indexOffset += mesh->m_verticesCount;
        faceCount += mesh->m_facesCount;
    }

#if 0
    unsigned int sum = 0;
    const auto& mesh1 = m_meshes[0];
    const auto& mesh2 = m_meshes[1];
    std::vector<owl::vec2ui> intTriPairs;
    for (unsigned int i = 0; i < mesh1->m_facesCount; i++)
    {
        for (unsigned int j = 0; j < mesh2->m_facesCount; j++)
        {
            const auto index1a = mesh1->m_indices[i * 3 + 0];
            const auto index1b = mesh1->m_indices[i * 3 + 1];
            const auto index1c = mesh1->m_indices[i * 3 + 2];
            const auto index2a = mesh2->m_indices[j * 3 + 0];
            const auto index2b = mesh2->m_indices[j * 3 + 1];
            const auto index2c = mesh2->m_indices[j * 3 + 2];
            if (triangleIntersect(mesh1->m_vertices[index1a].m_position,
                                  mesh1->m_vertices[index1b].m_position,
                                  mesh1->m_vertices[index1c].m_position,
                                  mesh2->m_vertices[index2a].m_position,
                                  mesh2->m_vertices[index2b].m_position,
                                  mesh2->m_vertices[index2c].m_position))
            {
                sum++;
                intTriPairs.push_back({i, j});
            }
        }
    }
    // printf("sum: %u\n", sum);
    std::ofstream outfile;
    outfile.open("C://Users//Administrator//Projects//VisualStudio//oibvh//logs//tripair_log.txt");
    for (int i = 0; i < sum; i++)
    {
        outfile << "(" << intTriPairs[i].x << "," << intTriPairs[i].y << ")" << std::endl;
    }
#endif

    genAuxTriangles(m_vertices, m_indices, m_auxVertices, m_auxIndices);

    // create a context on the first device:
    m_context = owlContextCreate(nullptr, 1);
    OWLModule module = owlModuleCreate(m_context, deviceCode_ptx);

    // ##################################################################
    // set up all the *GEOMETRY* graph we want to render
    // ##################################################################

    // -------------------------------------------------------
    // declare geometry type
    // -------------------------------------------------------
    OWLVarDecl trianglesGeomVars[] = {//{ "m_indices",    OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, m_indices)},
                                      //{ "m_vertices",   OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, m_vertices)},
                                      {"m_indexPairs", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, m_indexPairs)},
                                      {"m_count", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, m_count)},
                                      {"m_faceCounts", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, m_faceCounts)},
                                      {"m_size", OWL_UINT, OWL_OFFSETOF(TrianglesGeomData, m_size)}};
    OWLGeomType trianglesGeomType =
        owlGeomTypeCreate(m_context, OWL_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars, 4);
    owlGeomTypeSetAnyHit(trianglesGeomType, 0, module, "AnyHit");

    OWLVarDecl auxTrianglesGeomVars[] = {
        //   { "m_indices",    OWL_BUFPTR, OWL_OFFSETOF(AuxTrianglesGeomData, m_indices)},
        //{ "m_vertices",   OWL_BUFPTR, OWL_OFFSETOF(AuxTrianglesGeomData, m_vertices)},
        {"m_indexPairs", OWL_BUFPTR, OWL_OFFSETOF(AuxTrianglesGeomData, m_indexPairs)},
        {"m_count", OWL_BUFPTR, OWL_OFFSETOF(AuxTrianglesGeomData, m_count)},
        {"m_faceCounts", OWL_BUFPTR, OWL_OFFSETOF(AuxTrianglesGeomData, m_faceCounts)},
        {"m_size", OWL_UINT, OWL_OFFSETOF(AuxTrianglesGeomData, m_size)}};
    OWLGeomType auxTrianglesGeomType =
        owlGeomTypeCreate(m_context, OWL_TRIANGLES, sizeof(AuxTrianglesGeomData), auxTrianglesGeomVars, 4);
    owlGeomTypeSetAnyHit(auxTrianglesGeomType, 0, module, "AuxAnyHit");

    // ##################################################################
    // set up all the *GEOMS* we want to run that code on
    // ##################################################################

    LOG("building geometries ...");

    // ------------------------------------------------------------------
    // triangle mesh
    // ------------------------------------------------------------------
    m_vertexBuffer = owlDeviceBufferCreate(m_context, OWL_FLOAT3, m_vertices.size(), m_vertices.data());
    m_indexBuffer = owlDeviceBufferCreate(m_context, OWL_UINT3, m_indices.size(), m_indices.data());

    m_auxVertexBuffer = owlDeviceBufferCreate(m_context, OWL_FLOAT3, m_auxVertices.size(), m_auxVertices.data());
    m_auxIndexBuffer = owlDeviceBufferCreate(m_context, OWL_UINT3, m_auxIndices.size(), m_auxIndices.data());
    m_indexPairsBuffer = owlDeviceBufferCreate(m_context, OWL_UINT2, 100000000, nullptr);
    unsigned int count = 0;
    m_countBuffer = owlDeviceBufferCreate(m_context, OWL_UINT, 1, &count);
    OWLBuffer faceCountsBuffer = owlDeviceBufferCreate(m_context, OWL_UINT, m_faceCounts.size(), m_faceCounts.data());

    m_trianglesGeom = owlGeomCreate(m_context, trianglesGeomType);

    owlTrianglesSetVertices(m_trianglesGeom, m_vertexBuffer, m_vertices.size(), sizeof(owl::vec3f), 0);
    owlTrianglesSetIndices(m_trianglesGeom, m_indexBuffer, m_indices.size(), sizeof(owl::vec3ui), 0);

    owlGeomSetBuffer(m_trianglesGeom, "m_indexPairs", m_indexPairsBuffer);
    owlGeomSetBuffer(m_trianglesGeom, "m_count", m_countBuffer);
    owlGeomSetBuffer(m_trianglesGeom, "m_faceCounts", faceCountsBuffer);
    owlGeomSet1ui(m_trianglesGeom, "m_size", m_faceCounts.size());

    m_auxTrianglesGeom = owlGeomCreate(m_context, auxTrianglesGeomType);

    owlTrianglesSetVertices(m_auxTrianglesGeom, m_auxVertexBuffer, m_auxVertices.size(), sizeof(owl::vec3f), 0);
    owlTrianglesSetIndices(m_auxTrianglesGeom, m_auxIndexBuffer, m_auxIndices.size(), sizeof(owl::vec3ui), 0);

    owlGeomSetBuffer(m_auxTrianglesGeom, "m_indexPairs", m_indexPairsBuffer);
    owlGeomSetBuffer(m_auxTrianglesGeom, "m_count", m_countBuffer);
    owlGeomSetBuffer(m_auxTrianglesGeom, "m_faceCounts", faceCountsBuffer);
    owlGeomSet1ui(m_auxTrianglesGeom, "m_size", m_faceCounts.size());

    // ------------------------------------------------------------------
    // the group/accel for that mesh
    // ------------------------------------------------------------------
    OWLGeom trianglesGeoms[2] = {m_trianglesGeom, m_auxTrianglesGeom};
    m_trianglesGroup = owlTrianglesGeomGroupCreate(m_context, 2, trianglesGeoms, OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    owlGroupBuildAccel(m_trianglesGroup);
    OWLGroup world = owlInstanceGroupCreate(m_context, 1, &m_trianglesGroup);
    owlGroupBuildAccel(world);

    // ##################################################################
    // set any hit and raygen program required for SBT
    // ##################################################################

    // -------------------------------------------------------
    // set up ray gen program
    // -------------------------------------------------------
    OWLVarDecl rayGenVars[] = {{"m_indices", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, m_indices)},
                               {"m_vertices", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, m_vertices)},
                               {"m_world", OWL_GROUP, OWL_OFFSETOF(RayGenData, m_world)},
                               {/* sentinel to mark end of list */}};

    // ----------- create object  ----------------------------
    m_rayGen = owlRayGenCreate(m_context, module, "RayGen", sizeof(RayGenData), rayGenVars, -1);

    // ----------- set variables  ----------------------------
    owlRayGenSetBuffer(m_rayGen, "m_indices", m_indexBuffer);
    owlRayGenSetBuffer(m_rayGen, "m_vertices", m_vertexBuffer);
    owlRayGenSetGroup(m_rayGen, "m_world", world);

    // ##################################################################
    // build *SBT* required to trace the groups
    // ##################################################################
    owlBuildPrograms(m_context);
    owlBuildPipeline(m_context);
    owlBuildSBT(m_context);
}

void OptixCollide::refit()
{
    unsigned int index = 0;
    for (auto& mesh : m_meshes)
    {
        for (auto& vertex : mesh->m_vertices)
        {
            m_vertices[index++] = {vertex.m_position.x, vertex.m_position.y, vertex.m_position.z};
        }
    }
    genAuxTriangles(m_vertices, m_indices, m_auxVertices, m_auxIndices);
    owlBufferUpload(m_vertexBuffer, m_vertices.data());
    owlTrianglesSetVertices(m_trianglesGeom, m_vertexBuffer, m_vertices.size(), sizeof(owl::vec3f), 0);
    owlBufferUpload(m_auxVertexBuffer, m_auxVertices.data());
    owlTrianglesSetVertices(m_auxTrianglesGeom, m_auxVertexBuffer, m_auxVertices.size(), sizeof(owl::vec3f), 0);
    owlGroupRefitAccel(m_trianglesGroup);
}

void OptixCollide::detect()
{
    // ##################################################################
    // now that everything is ready: launch it ....
    // ##################################################################
    // LOG("launching ...");
    unsigned int count = 0;
    owlBufferUpload(m_countBuffer, &count);
    float elapsedTime = kernelLaunch([&]() { owlRayGenLaunch2D(m_rayGen, m_indices.size(), 1); });
    if (m_detectTimes % 100 == 0)
    {
        LOG("Optix Detect time: " + std::to_string(elapsedTime) + "ms");
    }

    // LOG("done with launch");
    cudaMemcpy(&count, owlBufferGetPointer(m_countBuffer, 0), sizeof(unsigned int), cudaMemcpyDeviceToHost);

    m_indexPairs.resize(count);
    cudaMemcpy(m_indexPairs.data(),
               owlBufferGetPointer(m_indexPairsBuffer, 0),
               count * sizeof(owl::vec2ui),
               cudaMemcpyDeviceToHost);

    dumplicationRemove(m_indexPairs);
    if (m_detectTimes % 100 == 0)
    {
        LOG("optix collide count: " + std::to_string(m_indexPairs.size()));
        printf("\n");
    }
    m_convertDone = false;
    m_detectTimes++;
}