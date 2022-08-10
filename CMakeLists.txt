cmake_minimum_required (VERSION 3.17 FATAL_ERROR)

project(OCCA_TUTORIAL LANGUAGES CXX)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/occa" ${CMAKE_MODULE_PATH})

#
# Configure OCCA (REQUIRED)
#
add_subdirectory(occa)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} -O3 -Wno-unused-parameter")
else()
  set(CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} -O0 -g -Wno-unused-parameter")
endif()

add_subdirectory(01_Introduction)
add_subdirectory(02_Loops)
add_subdirectory(03_Reduction)
add_subdirectory(04_Streams)
add_subdirectory(05_Inline_Kernels)

if (OCCA_HIP_ENABLED)
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm/)

  # Find HIP package
  find_package(HIP REQUIRED)

  add_subdirectory(06_Native_Interop)
endif()
