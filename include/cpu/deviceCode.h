// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <owl/owl.h>
#include <owl/common/math/vec.h>

/* variables for the triangle mesh geometry */
struct TrianglesGeomData
{
    ///*! array/buffer of vertex indices */
    // owl::vec3ui* m_indices;
    ///*! array/buffer of vertex positions */
    // owl::vec3f* m_vertices;
    owl::vec2ui* m_indexPairs;
    unsigned int* m_count;
};

/* variables for the auxilury triangle mesh geometry */
struct AuxTrianglesGeomData
{
    ///*! array/buffer of vertex indices */
    // owl::vec3ui* m_indices;
    ///*! array/buffer of vertex positions */
    // owl::vec3f* m_vertices;
    owl::vec2ui* m_indexPairs;
    unsigned int* m_count;
};

/* variables for the ray generation program */
struct RayGenData
{
    owl::vec3ui* m_indices;
    owl::vec3f* m_vertices;
    OptixTraversableHandle m_world;
};
