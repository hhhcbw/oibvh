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
    owl::vec2ui* m_indexPairs;
    unsigned int* m_count;
    unsigned int m_size;
    unsigned int* m_faceCounts;
};

/* variables for the auxilury triangle mesh geometry */
struct AuxTrianglesGeomData
{
    owl::vec2ui* m_indexPairs;
    unsigned int* m_count;
    unsigned int m_size;
    unsigned int* m_faceCounts;
};

/* variables for the ray generation program */
struct RayGenData
{
    owl::vec3ui* m_indices;
    owl::vec3f* m_vertices;
    OptixTraversableHandle m_world;
};
