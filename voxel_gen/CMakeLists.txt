cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(voxel_gen)

find_package(pybind11 REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

add_library(voxel_generator SHARED voxel_generator.cc)
target_include_directories(voxel_generator PRIVATE ${pybind11_INCLUDE_DIRS})
set_property(TARGET voxel_generator PROPERTY CXX_STANDARD 14)
set_target_properties(voxel_generator PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
target_link_libraries(voxel_generator pybind11::module -Wl,--no-as-needed)
