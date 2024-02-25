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

__device__ inline int getMeshID(int faceIndex, unsigned int size, unsigned int* faceCounts)
{
    int l = 0;
    int r = size - 1;
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

OPTIX_RAYGEN_PROGRAM(RayGen)()
{
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const owl::vec2i pixelID = owl::getLaunchIndex();
    owl::vec3ui indexes = self.m_indices[pixelID.x];
    owl::vec3f v[3];
    v[0] = self.m_vertices[indexes.x];
    v[1] = self.m_vertices[indexes.y];
    v[2] = self.m_vertices[indexes.z];
    // printf("%u %u %u %u\n", pixelID.x, indexes.x, indexes.y, indexes.z);

    owl::Ray ray;
    owl::vec3f prd;
    for (int i = 0; i < 3; i++)
    {
        ray.origin = v[i];
        ray.direction = owl::normalize(v[(i + 1) % 3] - v[i]);
        ray.tmax = owl::length(v[(i + 1) % 3] - v[i]) + 0.0001f;
        owl::traceRay(self.m_world, ray, prd);
    }

    //owl::vec3f norm = owl::normalize(cross(v[1] - v[0], v[2] - v[0]));
    //for (int i = 0; i < 3; i++)
    //{
    //    ray.origin = v[i] - 0.000001f * norm;
    //    ray.direction = norm;
    //    ray.tmax = 0.0000011f;
    //    owl::traceRay(self.m_world, ray, prd);
    //}
}

OPTIX_ANY_HIT_PROGRAM(AnyHit)()
{
    const TrianglesGeomData& self = owl::getProgramData<TrianglesGeomData>();
    const int primID = optixGetPrimitiveIndex();
    const owl::vec2i pixelID = owl::getLaunchIndex();
    const int meshID1 = getMeshID(primID, self.m_size, self.m_faceCounts);
    const int meshID2 = getMeshID(pixelID.x, self.m_size, self.m_faceCounts);
    if (meshID1 == meshID2)
	{
		return;
	}
    const unsigned int idx = atomicAdd(self.m_count, 1);
    // printf("idx:%u %u %u\n", idx, primID, pixelID.x);
    self.m_indexPairs[idx] = owl::vec2ui{pixelID.x, primID};
}

OPTIX_ANY_HIT_PROGRAM(AuxAnyHit)()
{
    const AuxTrianglesGeomData& self = owl::getProgramData<AuxTrianglesGeomData>();
    const int primID = optixGetPrimitiveIndex();
    const owl::vec2i pixelID = owl::getLaunchIndex();
    const int meshID1 = getMeshID(primID / 3, self.m_size, self.m_faceCounts);
    const int meshID2 = getMeshID(pixelID.x, self.m_size, self.m_faceCounts);
    if (meshID1 == meshID2)
    {
        return;
    }

    // check hit point is on the true triangle
    const float2 barycentrics = optixGetTriangleBarycentrics();
    if (1.0f - barycentrics.y >= 2.0f * barycentrics.x + 0.0001f ||
        1.0f - barycentrics.y <= 2.0f * barycentrics.x - 0.0001f)
        return;

    const unsigned int idx = atomicAdd(self.m_count, 1);
    // printf("idx:%u %u %u\n", idx, primID / 3, pixelID.x);
    self.m_indexPairs[idx] = owl::vec2ui{pixelID.x, primID / 3};
}
