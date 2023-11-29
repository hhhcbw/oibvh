#include "utils/utils.h"

std::ostream& operator<<(std::ostream& os, const glm::vec3& gvec3)
{
	return os << "(" << gvec3.x << ", " << gvec3.y << ", " << gvec3.z << ")";
}