cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(ukf_slam)

find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)

set(CNPY_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../mapping/cnpy/include)
set(CNPY_LIB_DIR ${CMAKE_SOURCE_DIR}/../mapping/cnpy/lib)
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CNPY_INCLUDE_DIR} ${PROJECT_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS} ${CNPY_LIB_DIR})
add_definitions(${PCL_DEFINITIONS})

SET(FLAG_A "-DUSE_RANSAC_PNP")
SET(FLAG_B "-DUSE_DOUBLE_PRECISION")

add_executable (ukf main.cpp)
target_link_libraries (ukf ${PCL_LIBRARIES} ${OpenCV_LIBS} cnpy z jsoncpp stdc++fs OpenMP::OpenMP_CXX ${FLAG_A} ${FLAG_B})
