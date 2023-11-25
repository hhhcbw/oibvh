cmake_minimum_required (VERSION 3.10)
project (lightPsep C CXX CUDA) 

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_INCLUDE_DIRS_PATH" ${CUDA_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MOUDLE_PATH: ${CMAKE_MODULE_PATH}")

find_package(OpenGL REQUIRED)
list(APPEND THIRD_LIBS OpenGL::GL)
find_package(assimp REQUIRED)
list (APPEND THIRD_INCLUDE_DIR ${ASSIMP_INCLUDE_DIRS})
list (APPEND THIRD_LIBS ${ASSIMP_LIBRARIES})
set (lightPsep_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (THIRD_BASE_DIR ${lightPsep_BASE_DIR}/third)
add_subdirectory(third)
list (APPEND THIRD_INCLUDE_DIR ${THIRD_BASE_DIR} ${GLAD_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${IMGUIGLFWOPENGL3_INCLUDE_DIR})
message (STATUS "third include directory: ${THIRD_INCLUDE_DIR}")
list(APPEND THIRD_LIBS ${GLAD_LIBRARIES} ${GLFW_LIBRARIES} ${IMGUIGLFWOPENGL3_LIBRARIES})

set (lightPsep_INCLUDE_DIR ${lightPsep_BASE_DIR}/include)
set (lightPsep_SRC_DIR ${lightPsep_BASE_DIR}/src)

aux_source_directory(${lightPsep_SRC_DIR} lightPsep_SRC)
aux_source_directory(${lightPsep_SRC_DIR}/utils lightPsep_SRC)
aux_source_directory(${lightPsep_SRC_DIR}/cuda lightPsep_SRC)
message(STATUS "lightPsep source files: ${lightPsep_SRC}")

add_executable(lightPsep ${lightPsep_SRC})
target_include_directories(lightPsep PUBLIC ${lightPsep_INCLUDE_DIR} ${THIRD_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
message(STATUS "third libs: ${THIRD_LIBS}")
target_link_libraries(lightPsep PUBLIC ${THIRD_LIBS})