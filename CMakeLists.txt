# 3.8+ for project(LANGUAGES CUDA)
# 3.9+ for OpenMP::OpenMP_CXX
# 3.10+ findopenmp gained support for language-specific components
# 3.11+ for CMake not to add -fopenmp to the nvcc flags
# 3.13+ for target_link_directories

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# set(
#     CMAKE_TOOLCHAIN_FILE
#     "${CMAKE_CURRENT_LIST_DIR}/cmake/toolchain.cmake"
#     FILEPATH
#     "Default toolchain"
# )

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.115.tar.gz"
    SHA1 "1b3f3addc801405769edbb7ebbe701223db3efa6"
    LOCAL
)

project(pangolin LANGUAGES CXX CUDA VERSION 0.1.0.0)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set_property(GLOBAL PROPERTY USE_FOLDERS OFF)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include("${CMAKE_CURRENT_LIST_DIR}/cmake/GetGitRevisionDescription.cmake")


get_git_head_revision(GIT_REFSPEC GIT_HASH)
git_local_changes(GIT_LOCAL_CHANGES)
message(STATUS GIT_REFSPEC=${GIT_REFSPEC})
message(STATUS GIT_HASH=${GIT_HASH})
message(STATUS GIT_LOCAL_CHANGES=${GIT_LOCAL_CHANGES})

option(USE_HUNTER "Turn on to enable using the hunter package manager" ON)
option(USE_OPENMP "compile with OpenMP support" ON)
option(USE_NUMA "compile with NUMA support" ON)
option(USE_CUSPARSE "compile with CUSparse" ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

if(USE_HUNTER)
  hunter_add_package(fmt)
endif()
find_package(fmt CONFIG REQUIRED)

if(USE_HUNTER)
  hunter_add_package(spdlog)
endif()
find_package(spdlog CONFIG REQUIRED)

if(USE_HUNTER)
  hunter_add_package(cub)
endif()
find_package(cub CONFIG REQUIRED)

find_package(OpenMP)

if (USE_NUMA)
find_package(NUMA)
endif(USE_NUMA)

set(PANGOLIN_CPP_HEADERS
    include/pangolin/bounded_buffer.hpp
    include/pangolin/config.hpp
    include/pangolin/cusparse.hpp
    include/pangolin/dag_lowertriangular_csr.hpp
    include/pangolin/edge.hpp
    include/pangolin/init.hpp
    include/pangolin/logger.hpp
    include/pangolin/numa.hpp
    include/pangolin/pangolin.hpp
    include/pangolin/par_graph.hpp
    include/pangolin/utilities.hpp
    include/pangolin/allocator/cuda_managed.hpp
    include/pangolin/allocator/cuda_zero_copy.hpp
    include/pangolin/dense/cuda_managed_vector.hpp
    include/pangolin/dense/cuda_zero_copy_vector.hpp
    include/pangolin/file/edge_list_file.hpp
    include/pangolin/generator/complete.hpp
    include/pangolin/sparse/dag2019.hpp
    include/pangolin/sparse/gpu_csr.hpp
    include/pangolin/sparse/gpu_csr-impl.hpp
    include/pangolin/sparse/csr_coo.hpp
    include/pangolin/sparse/coo-impl.hpp
    include/pangolin/sparse/unified_memory_csr.hpp
    include/pangolin/reader/bel_reader.hpp
    include/pangolin/reader/edge_list_reader.hpp
    include/pangolin/reader/gc_tsv_reader.hpp
)

set(PANGOLIN_CU_HEADERS
  include/pangolin/atomic_add.cuh
  include/pangolin/sparse/cusparse_csr.hu
  include/pangolin/dense/vector.cuh
  include/pangolin/dense/vector-impl.hu
)
add_subdirectory(include)


# pangolin header-only library
add_library(pangolin INTERFACE)
add_library(pangolin::pangolin ALIAS pangolin)

# need to add the arch flags to the device link step as well
# there is no way to do this through targets, so we do it
# the old fashioned way
# include PTX for at least dynamic parallelism, plus binaries for Pascal, Volta, and Turing as supported
# -arch specifies the class of virtual architecture the source must be compiled for 
# -compute specifies the PTX (compute) and SASS (sm) embedded in the binary
if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 10)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -code=compute_35,sm_61,sm_70,sm_75")
elseif (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 9)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -code=compute_35,sm_61,sm_70")
elseif (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 8)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -code=compute_35,sm_61")
else()
set(CMAKE_CUDA_FLAGS "-arch=compute_35") # -code implicitly is also compute_35
endif()

## Add CUDA flags
target_compile_options(
  pangolin
  INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --Wno-deprecated-gpu-targets;
    --expt-extended-lambda;
    -Xcompiler=-Wall;
    -Xcompiler=-Wextra;
    -Xcompiler=-Wcast-align;
    -Xcompiler=-Wstrict-aliasing;
    -Xcompiler=-Wpointer-arith;
    -Xcompiler=-Winit-self;
    -Xcompiler=-Wswitch-enum;
    -Xcompiler=-Wfloat-equal;
    -Xcompiler=-Wvla;
    -Xcompiler=-Wshadow;
  >
)

## Add CXX Flags
target_compile_options(
  pangolin
  INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:
    -Wall;
    -Wextra;
    -Wcast-align;
    -Wstrict-aliasing;
    -Wpointer-arith;
    -Winit-self;
    -Wswitch-enum;
    -Wfloat-equal;
    -Wundef;
    -Wvla;
    -Wshadow;
    -Wformat=2;
    -Wconversion;
    -Wpedantic;
  >
)

# target_include_directories(pangolin INTERFACE include/)

#set_target_properties(pangolin PROPERTIES
#  CUDA_SEPARABLE_COMPILATION ON
#  CUDA_RESOLVE_DEVICE_SYMBOLS ON
#)


if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(
    pangolin
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-G>
  )
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(
    pangolin
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
endif()




## Add OpenMP Flags
if(OPENMP_FOUND)
  target_compile_definitions(pangolin INTERFACE -DUSE_OPENMP)
  target_link_libraries(pangolin INTERFACE OpenMP::OpenMP_CXX)
  target_compile_options(
    pangolin
    INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
  )
endif()

#include(GenerateExportHeader)
#generate_export_header(pangolin)

set(bin_install_dir "bin")
set(include_install_dir "include")
set(lib_install_dir "lib")
set(config_install_dir "${lib_install_dir}/cmake/${PROJECT_NAME}")
set(docs_install_dir "docs")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

##  Handle the header file with pangolin version info
# message(STATUS "${PROJECT_SOURCE_DIR}/src/configure.hpp.in" " -> " "${PROJECT_BINARY_DIR}/include/pangolin/configure.hpp")
# configure_file (
#     "${PROJECT_SOURCE_DIR}/src/configure.hpp.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/include/pangolin/configure.hpp"
# )
message(STATUS "${PROJECT_SOURCE_DIR}/src/configure.hpp.in" " -> " "${PROJECT_BINARY_DIR}/include/pangolin/configure.hpp")
configure_file (
    "${PROJECT_SOURCE_DIR}/include/pangolin/configure.hpp.in"
    "${PROJECT_SOURCE_DIR}/include/pangolin/configure.hpp"
)
# # install the pangolin version file
# install(
#   FILES "${CMAKE_CURRENT_BINARY_DIR}/include/pangolin/configure.hpp"
#   DESTINATION include/pangolin
# )
# include the binary dir during build or the install dir during install
# target_include_directories(pangolin INTERFACE
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/pangolin> # fake that "configure.hpp" is in same dir as pangolin.hpp
#   $<INSTALL_INTERFACE:include>
# )
target_include_directories(pangolin INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> 
  $<INSTALL_INTERFACE:include>
)
target_include_directories(pangolin INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/thirdparty> 
  $<INSTALL_INTERFACE:thirdparty>
)

include(CMakePackageConfigHelpers)
# Create the CMake version file
write_basic_package_version_file(
  "${version_config}"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# create the projectConfig.cmake file
configure_package_config_file(
    "cmake/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
)

# Install the include files
install(
  DIRECTORY
    include/pangolin
  DESTINATION
    ${include_install_dir}
  COMPONENT
    Devel
)

# install the pangolinConfig and pangolinVersionConfig.cmake files
install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
)

# install libraries
install(
    TARGETS pangolin
    EXPORT "${targets_export_name}"
    INCLUDES DESTINATION "${include_install_dir}"
    LIBRARY DESTINATION "${lib_install_dir}"
    ARCHIVE DESTINATION "${lib_install_dir}"
    RUNTIME DESTINATION "${bin_install_dir}"
)

install(
    EXPORT "${targets_export_name}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
)



## Add include directories
target_include_directories(pangolin SYSTEM INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(pangolin INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)



target_link_libraries(pangolin INTERFACE spdlog::spdlog)
target_link_libraries(pangolin INTERFACE cub::cub)
target_link_libraries(pangolin INTERFACE nvgraph)
target_link_libraries(pangolin INTERFACE nvToolsExt)
target_link_libraries(pangolin INTERFACE nvidia-ml)
if (USE_CUSPARSE)
  target_compile_definitions(pangolin INTERFACE -DPANGOLIN_USE_CUSPARSE=1)
  target_link_libraries(pangolin INTERFACE cusparse)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang or AppleClang
  # set_target_properties(pangolin PROPERTIES LINK_FLAGS -Wl,--no-undefined)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  # set_target_properties(pangolin PROPERTIES LINK_FLAGS -Wl,--no-undefined)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()

# Request that pangolin and anything requiring it be built with -std=c++11
target_compile_features(pangolin INTERFACE cxx_std_11)
# set_target_properties(pangolin PROPERTIES
#   CUDA_STANDARD 11
#   CUDA_STANDARD_REQUIRED ON
#   CUDA_EXTENSIONS OFF
#   CXX_STANDARD 11
#   CXX_STANDARD_REQUIRED ON
#   CXX_EXTENSIONS OFF
# )

# add the binary dir where the version file is
target_include_directories(pangolin INTERFACE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Setting verbose build during Debug")
  set(CMAKE_VERBOSE_MAKEFILE ON)
  target_compile_definitions(pangolin INTERFACE -DSPDLOG_TRACE_ON)
  target_compile_definitions(pangolin INTERFACE -DSPDLOG_DEBUG_ON)
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_definitions(pangolin INTERFACE -DNDEBUG)
  target_compile_definitions(pangolin INTERFACE -DSPDLOG_DEBUG_ON)
endif()

# Add a target to generate API documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(docs
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
  # install docs if they exist
  install(
    DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/docs/
    DESTINATION
      docs
    COMPONENT
      docs
    OPTIONAL
)
endif(DOXYGEN_FOUND)

enable_testing() # "this command should be in the source directory root for CTest to find the test file"
add_subdirectory(test)
