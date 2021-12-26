cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(map_create)

find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(CNPY_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/cnpy/include)
set(CNPY_LIB_DIR ${CMAKE_SOURCE_DIR}/cnpy/lib)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CNPY_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS} ${CNPY_LIB_DIR})
add_definitions(${PCL_DEFINITIONS})

add_executable (cmap create_map.cpp)
target_link_libraries (cmap ${PCL_LIBRARIES} ${OpenCV_LIBS} cnpy z jsoncpp stdc++fs OpenMP::OpenMP_CXX)
