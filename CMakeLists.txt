cmake_minimum_required(VERSION 3.10)
project(geoik-velctrl VERSION 0.1.0)

# Project description
set(PROJECT_DESCRIPTION "Geometric IK-based velocity control server for Franka robot")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization level to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Set CMAKE_PREFIX_PATH to find your libfranka installation
# You can override this by passing -DFRANKA_INSTALL_PATH=/path/to/franka/install
set(FRANKA_INSTALL_PATH "$ENV{HOME}/ws/franka/install" CACHE PATH "Path to Franka installation directory")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${FRANKA_INSTALL_PATH}")

# Find required packages
find_package(Franka REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ruckig REQUIRED)

# Common utilities library (Franka examples)
add_library(examples_common src/examples_common.cpp)
target_link_libraries(examples_common PUBLIC Franka::Franka)
target_include_directories(examples_common PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${EIGEN3_INCLUDE_DIR}
)

# GeoFIK geometric IK library
add_library(geofik src/geofik.cpp)
target_include_directories(geofik PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${EIGEN3_INCLUDE_DIR}
)

# Weighted IK optimization library
add_library(weighted_ik src/weighted_ik.cpp)
target_link_libraries(weighted_ik PUBLIC geofik)
target_include_directories(weighted_ik PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${EIGEN3_INCLUDE_DIR}
)

# Main velocity control server executable
add_executable(franka_velocity_server src/franka_velocity_server.cpp)

# Link libraries
target_link_libraries(franka_velocity_server
    Franka::Franka
    Threads::Threads
    examples_common
    weighted_ik
    geofik
    ruckig::ruckig
    ${EIGEN3_LIBRARIES}
)

# Include directories
target_include_directories(franka_velocity_server PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Set runtime path for shared libraries
set_target_properties(franka_velocity_server PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Print build configuration
message(STATUS "")
message(STATUS "=== geoik-velctrl Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Franka path: ${FRANKA_INSTALL_PATH}")
message(STATUS "Eigen3 include: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "===================================")
message(STATUS "")