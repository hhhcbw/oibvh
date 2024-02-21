#pragma once
// public owl node-graph API
#include "owl/owl.h"
// our device-side data structures
#include "cpu/deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <vector>

#define LOG(message)                                            \
    std::cout << OWL_TERMINAL_BLUE;                             \
    std::cout << "#owl.sample(main): " << message << std::endl; \
    std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
    std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
    std::cout << "#owl.sample(main): " << message << std::endl; \
    std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

std::vector<owl::vec3f> vertices = {
    {-0.5f, -0.5f, 0.0f},   // 0
    {0.5f, -0.5f, 0.0f},    // 1
    {0.0f, 0.5f, 0.0f},     // 2
    {0.0f, 0.0f, -0.5f},    // 3
    {0.0f, 0.0f, 0.5f},     // 4
    {0.5f, 0.0f, 0.0f},     // 5
    {-0.25f, 0.0f, 0.0f},   // 6
    {-0.25f, 0.0f, -0.5f},  // 7
    {-0.25f, 0.0f, 0.5f},   // 8
    {0.25f, 0.0f, 0.5f},    // 9
    {0.5f, 0.5f, 0.5f},     // 10
    {-0.5f, 0.5f, 0.5f},    // 11
    {0.0f, 0.0f, 0.0f},     // 12
    {0.5f, 0.0f, 0.0f},     // 13
    {0.5f, 0.5f, 0.0f},     // 14
    {0.0f, -0.5f, 0.5f},    // 15
    {-0.25f, -0.25f, 0.0f}, // 16
    {0.25f, -0.25f, 0.0f},  // 17
    {0.0f, 0.0f, 0.01f},    // 18
    {1.0f, 0.0f, 0.01f},    // 19
    {0.5f, 1.0f, 0.01f}     // 20
};

std::vector<owl::vec3ui> indices = {{0, 1, 2}, {18, 19, 20}};

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

void OptixCollide()
{
    // create a context on the first device:
    OWLContext context = owlContextCreate(nullptr, 1);
    OWLModule module = owlModuleCreate(context, deviceCode_ptx);

    // ##################################################################
    // set up all the *GEOMETRY* graph we want to render
    // ##################################################################

    // -------------------------------------------------------
    // declare geometry type
    // -------------------------------------------------------
    OWLVarDecl trianglesGeomVars[] = {//{ "m_indices",    OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, m_indices)},
                                      //{ "m_vertices",   OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, m_vertices)},
                                      {"m_indexPairs", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, m_indexPairs)},
                                      {"m_count", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, m_count)}};
    OWLGeomType trianglesGeomType =
        owlGeomTypeCreate(context, OWL_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars, 2);
    owlGeomTypeSetAnyHit(trianglesGeomType, 0, module, "AnyHit");

    OWLVarDecl auxTrianglesGeomVars[] = {
        //   { "m_indices",    OWL_BUFPTR, OWL_OFFSETOF(AuxTrianglesGeomData, m_indices)},
        //{ "m_vertices",   OWL_BUFPTR, OWL_OFFSETOF(AuxTrianglesGeomData, m_vertices)},
        {"m_indexPairs", OWL_BUFPTR, OWL_OFFSETOF(AuxTrianglesGeomData, m_indexPairs)},
        {"m_count", OWL_BUFPTR, OWL_OFFSETOF(AuxTrianglesGeomData, m_count)}};
    OWLGeomType auxTrianglesGeomType =
        owlGeomTypeCreate(context, OWL_TRIANGLES, sizeof(AuxTrianglesGeomData), auxTrianglesGeomVars, 2);
    owlGeomTypeSetAnyHit(auxTrianglesGeomType, 0, module, "AuxAnyHit");

    // ##################################################################
    // set up all the *GEOMS* we want to run that code on
    // ##################################################################

    LOG("building geometries ...");

    // ------------------------------------------------------------------
    // triangle mesh
    // ------------------------------------------------------------------
    OWLBuffer vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, vertices.size(), vertices.data());
    OWLBuffer indexBuffer = owlDeviceBufferCreate(context, OWL_UINT3, indices.size(), indices.data());

    std::vector<owl::vec3f> auxVertices;
    std::vector<owl::vec3ui> auxIndices;
    genAuxTriangles(vertices, indices, auxVertices, auxIndices);
    OWLBuffer auxVertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, auxVertices.size(), auxVertices.data());
    OWLBuffer auxIndexBuffer = owlDeviceBufferCreate(context, OWL_UINT3, auxIndices.size(), auxIndices.data());
    OWLBuffer indexPairsBuffer = owlDeviceBufferCreate(context, OWL_UINT2, 100000, nullptr);
    unsigned int count = 0;
    OWLBuffer countBuffer = owlDeviceBufferCreate(context, OWL_UINT, 1, &count);

    OWLGeom trianglesGeom = owlGeomCreate(context, trianglesGeomType);

    owlTrianglesSetVertices(trianglesGeom, vertexBuffer, vertices.size(), sizeof(owl::vec3f), 0);
    owlTrianglesSetIndices(trianglesGeom, indexBuffer, indices.size(), sizeof(owl::vec3ui), 0);

    // owlGeomSetBuffer(trianglesGeom, "m_indices", indexBuffer);
    // owlGeomSetBuffer(trianglesGeom, "m_vertices", vertexBuffer);
    owlGeomSetBuffer(trianglesGeom, "m_indexPairs", indexPairsBuffer);
    owlGeomSetBuffer(trianglesGeom, "m_count", countBuffer);

    OWLGeom auxTrianglesGeom = owlGeomCreate(context, auxTrianglesGeomType);

    owlTrianglesSetVertices(auxTrianglesGeom, auxVertexBuffer, auxVertices.size(), sizeof(owl::vec3f), 0);
    owlTrianglesSetIndices(auxTrianglesGeom, auxIndexBuffer, auxIndices.size(), sizeof(owl::vec3ui), 0);

    // owlGeomSetBuffer(auxTrianglesGeom, "m_indices", auxIndexBuffer);
    // owlGeomSetBuffer(auxTrianglesGeom, "m_vertices", auxVertexBuffer);
    owlGeomSetBuffer(auxTrianglesGeom, "m_indexPairs", indexPairsBuffer);
    owlGeomSetBuffer(auxTrianglesGeom, "m_count", countBuffer);

    // ------------------------------------------------------------------
    // the group/accel for that mesh
    // ------------------------------------------------------------------
    OWLGeom trianglesGeoms[2] = {trianglesGeom, auxTrianglesGeom};
    OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(context, 2, trianglesGeoms, OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    owlGroupBuildAccel(trianglesGroup);
    OWLGroup world = owlInstanceGroupCreate(context, 1, &trianglesGroup);
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
    OWLRayGen rayGen = owlRayGenCreate(context, module, "RayGen", sizeof(RayGenData), rayGenVars, -1);

    // ----------- set variables  ----------------------------
    owlRayGenSetBuffer(rayGen, "m_indices", indexBuffer);
    owlRayGenSetBuffer(rayGen, "m_vertices", vertexBuffer);
    owlRayGenSetGroup(rayGen, "m_world", world);

    // ##################################################################
    // build *SBT* required to trace the groups
    // ##################################################################
    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
    // owlBuildSBT(context, OWL_SBT_HITGROUPS);
    // owlBuildSBT(context, OWL_SBT_RAYGENS);

    // ##################################################################
    // now that everything is ready: launch it ....
    // ##################################################################

    LOG("launching ...");
    owlRayGenLaunch2D(rayGen, indices.size(), 1);

    LOG("done with launch");
    cudaMemcpy(&count, owlBufferGetPointer(countBuffer, 0), sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::vector<owl::vec2ui> indexPairs(100000);
    cudaMemcpy(indexPairs.data(),
               owlBufferGetPointer(indexPairsBuffer, 0),
               count * sizeof(owl::vec2ui),
               cudaMemcpyDeviceToHost);

    // LOG("Update vertices...");
    // for (auto& vertex : vertices)
    //{
    //	vertex += 0.1f;
    // }
    // owlBufferUpload(vertexBuffer, vertices.data());
    // owlTrianglesSetVertices(trianglesGeom, vertexBuffer,
    //	vertices.size(), sizeof(owl::vec3f), 0);
    // owlTrianglesSetIndices(trianglesGeom, indexBuffer,
    //	indices.size(), sizeof(owl::vec3ui), 0);
    // owlTrianglesSetVertices(auxTrianglesGeom, auxVertexBuffer, auxVertices.size(), sizeof(owl::vec3f), 0);
    // owlTrianglesSetIndices(auxTrianglesGeom, auxIndexBuffer,
    //	auxIndices.size(), sizeof(owl::vec3ui), 0);
    // owlGroupRefitAccel(trianglesGroup);

    // LOG("launching again...");
    // owlRayGenLaunch2D(rayGen, indices.size(), 1);
    //  ##################################################################
    //  and finally, clean up
    //  ##################################################################

    LOG("destroying devicegroup ...");
    owlContextDestroy(context);

    LOG_OK("seems all went OK; app is done, this should be the last output ...");
}