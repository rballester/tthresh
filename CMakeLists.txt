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

# Boost for multiprecision
# Windows only, Unix has built-in 128bit floating point ops
IF (WIN32)
find_package(Boost)
ENDIF()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Eigen
set(EIGEN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external)

# tthresh
file(GLOB sources src/*.cpp)
add_executable(tthresh ${sources})
include_directories(${EIGEN_INCLUDE_DIR} ${ZLIB_INCLUDE_DIRS})
if (WIN32)
    if (Boost_FOUND)
        include_directories(${Boost_INCLUDE_DIR})
        add_definitions(-DHAS_BOOST=1)
    else()
        add_definitions(-DHAS_BOOST=0)
    endif()
endif()
target_link_libraries(tthresh ${ZLIB_LIBRARY})
add_definitions(-std=c++11)
