##
# @file    CMakeLists.txt
# @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
#          Jiri Jaros <jarosjir@fit.vutbr.cz>
#
# @brief   Course: PPP 2019/2020 - Project 1
#          CMake build system project specifications.
#
# @date    2020-03-02
##

cmake_minimum_required(VERSION 2.8)
project(ppp_proj01)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS    "${CMAKE_CXX_LINK_FLAGS}    ${MPI_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH})

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 COMPONENTS C HL REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS})

add_compile_options(-Wall -Wextra -pedantic)

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

if(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xhost -inline-forceinline")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

file(GLOB_RECURSE HEADERS_LIST "*.h" "*.hpp")

set(SRCS_LIST
    main.cpp
    material_properties.cpp
    simulation_properties.cpp
    base_heat_solver.cpp
    sequential_heat_solver.cpp
    parallel_heat_solver.cpp
)

set(CMAKE_CXX_STANDARD 11)
add_executable(${PROJECT_NAME} ${SRCS_LIST} ${HEADERS_LIST})
target_link_libraries(${PROJECT_NAME} PRIVATE ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} ${MPI_LIBRARIES})

add_executable(data_generator data_generator.cpp)
target_link_libraries(data_generator PRIVATE ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

add_custom_target(test DEPENDS ${PROJECT_NAME} data_generator)
add_custom_command(TARGET test POST_BUILD
    COMMAND ./data_generator -N 256 -o test_input.h5
    COMMAND ./${PROJECT_NAME} -n 100 -m 0 -w 10 -i test_input.h5 -o result.h5)

if(CMAKE_VERSION VERSION_GREATER 3.2.0)
    add_custom_target(pack COMMAND
        ${CMAKE_COMMAND} -E tar "cfv" "${CMAKE_CURRENT_BINARY_DIR}/xlogin00.zip" --format=zip
            "parallel_heat_solver.h"
            "parallel_heat_solver.cpp"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
