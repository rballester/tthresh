cmake_minimum_required(VERSION 2.8.12)

project(tthresh CXX)
#set(CMAKE_CXX_STANDARD 11)

file(GLOB sources *.cpp)
#message(STATUS "sources = ${sources}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${PROJECT_SOURCE_DIR}/)
add_executable(tthresh ${sources})
#target_link_libraries()
add_definitions(-std=c++11)

