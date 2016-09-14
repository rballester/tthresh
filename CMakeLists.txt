cmake_minimum_required(VERSION 2.8.12)

project(tthresh CXX)
#set(CMAKE_CXX_STANDARD 11)

file(GLOB sources *.cpp)
#message(STATUS "sources = ${sources}")

include_directories(${PROJECT_SOURCE_DIR}/eigen)
add_executable(tthresh ${sources})
#target_link_libraries()
add_definitions(-std=c++11)

