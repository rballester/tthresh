cmake_minimum_required(VERSION 2.8.12)

project(tthresh CXX)
#set(CMAKE_CXX_STANDARD 11)

# External dependencies

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Eigen
set(EIGEN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external)

# zlib
set(SKIP_INSTALL_ALL on)
set(ZLIB_TESTS OFF CACHE BOOL "Build zlib tests")  # Don't build zlib tests
set(ZLIB_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/zlib)
add_subdirectory(external/zlib ${ZLIB_BINARY_DIR})
set(ZLIB_LIBRARY zlibstatic)
set(ZLIB_INCLUDE_DIRS zlib ${ZLIB_BINARY_DIR})

# tthresh
file(GLOB sources src/*.cpp)
add_executable(tthresh ${sources})
include_directories(${EIGEN_INCLUDE_DIR} ${ZLIB_INCLUDE_DIRS})
target_link_libraries(tthresh ${ZLIB_LIBRARY})
add_definitions(-std=c++11)
