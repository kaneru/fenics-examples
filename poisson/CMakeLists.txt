set(CMAKE_CXX_COMPILER "/usr/bin/g++")

cmake_minimum_required(VERSION 2.8)

project(demo_poisson)

cmake_policy(SET CMP0004 OLD)

find_package(DOLFIN)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Choose the type of build, options are:
        Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

add_definitions(${DOLFIN_CXX_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}")
include_directories(${DOLFIN_INCLUDE_DIRS} ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})
add_executable(demo_poisson main.cpp Poisson.cpp Error.cpp)
target_link_libraries(demo_poisson ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES})
