cmake_minimum_required(VERSION 3.0.0)
project(AuxEvaSystem)
set ( CMAKE_CUDA_STANDARD 14 )
 

enable_language(CUDA)
#
include(cmake/cmake.cmake)

enable_testing()


add_subdirectory(src)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
