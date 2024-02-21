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

#include "../include/cpu/deviceCode.h"
#include <optix_device.h>

OPTIX_RAYGEN_PROGRAM(RayGen)()
{
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const owl::vec2i pixelID = owl::getLaunchIndex();
    //printf("%d\n", pixelID.x);
    owl::vec3ui indexes = self.m_indices[pixelID.x];
    owl::vec3f v[3];
    v[0] = self.m_vertices[indexes.x];
    v[1] = self.m_vertices[indexes.y];
    v[2] = self.m_vertices[indexes.z];

    owl::Ray ray;
    owl::vec3f prd;
    for (int i = 0; i < 3; i++)
    {
        ray.origin = v[i];
        ray.direction = owl::normalize(v[(i + 1) % 3] - v[i]);
        owl::traceRay(self.m_world, ray, prd);
    }

    owl::vec3f norm = owl::normalize(cross(v[1] - v[0], v[2] - v[0]));
    for (int i = 0; i < 3; i++)
    {
        ray.origin = v[i] - 0.000001f * norm;
        ray.direction = norm;
        owl::traceRay(self.m_world, ray, prd);
    }
}

OPTIX_ANY_HIT_PROGRAM(AnyHit)()
{
    const TrianglesGeomData& self = owl::getProgramData<TrianglesGeomData>();
    const int primID = optixGetPrimitiveIndex();
    const owl::vec2i pixelID = owl::getLaunchIndex();
    const unsigned int idx = atomicAdd(self.m_count, 1);
    //printf("idx:%u %u %u\n", idx, primID, pixelID.x);
    self.m_indexPairs[idx] = owl::vec2ui{pixelID.x, primID};
}

OPTIX_ANY_HIT_PROGRAM(AuxAnyHit)()
{
    const AuxTrianglesGeomData& self = owl::getProgramData<AuxTrianglesGeomData>();
    const int primID = optixGetPrimitiveIndex();
    const owl::vec2i pixelID = owl::getLaunchIndex();

    // check hit point is on the true triangle
    const float2 barycentrics = optixGetTriangleBarycentrics();
    if (1.0f - barycentrics.y >= 2.0f * barycentrics.x + 0.0001f ||
        1.0f - barycentrics.y <= 2.0f * barycentrics.x - 0.0001f)
        return;

    const unsigned int idx = atomicAdd(self.m_count, 1);
    //printf("idx:%u %u %u\n", idx, primID / 3, pixelID.x);
    self.m_indexPairs[idx] = owl::vec2ui{pixelID.x, primID / 3};
}

