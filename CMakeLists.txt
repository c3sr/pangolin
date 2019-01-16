cmake_minimum_required(VERSION 3.12 FATAL_ERROR)


# set(
#     CMAKE_TOOLCHAIN_FILE
#     "${CMAKE_CURRENT_LIST_DIR}/cmake/toolchain.cmake"
#     FILEPATH
#     "Default toolchain"
# )

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.55.tar.gz"
    SHA1 "da31d97da47cdde05f958786b6729ba9b773bb17"
    LOCAL
)

project(pangolin LANGUAGES CXX CUDA VERSION 0.0.1)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include("${CMAKE_CURRENT_LIST_DIR}/cmake/GetGitRevisionDescription.cmake")


get_git_head_revision(GIT_REFSPEC GIT_HASH)
git_local_changes(GIT_LOCAL_CHANGES)
message(STATUS GIT_REFSPEC=${GIT_REFSPEC})
message(STATUS GIT_HASH=${GIT_HASH})
message(STATUS GIT_LOCAL_CHANGES=${GIT_LOCAL_CHANGES})

option(USE_HUNTER "Turn on to enable using the hunteger package manager" ON)
option(USE_OPENMP "compile with OpenMP support" ON)

project(pangolin LANGUAGES C CXX CUDA VERSION 0.1.0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

macro(tri_include_directories)
  target_include_directories(pangolin32 ${ARGN})
  target_include_directories(pangolin64 ${ARGN})
endmacro()
macro(tri_link_libraries)
  target_link_libraries(pangolin32 ${ARGN})
  target_link_libraries(pangolin64 ${ARGN})
endmacro()
macro(set_tri_properties)
  set_target_properties(pangolin32 ${ARGN})
  set_target_properties(pangolin64 ${ARGN})
endmacro()
macro(tri_compile_features)
  target_compile_features(pangolin32 ${ARGN})
  target_compile_features(pangolin64 ${ARGN})
endmacro()
macro(tri_compile_definitions)
  target_compile_definitions(pangolin32 ${ARGN})
  target_compile_definitions(pangolin64 ${ARGN})
endmacro()
macro(pangolin_compile_options)
  target_compile_options(pangolin32 ${ARGN})
  target_compile_options(pangolin64 ${ARGN})
endmacro()

if(USE_HUNTER)
  hunter_add_package(fmt)
endif()
find_package(fmt CONFIG REQUIRED)

hunter_add_package(spdlog)
find_package(spdlog CONFIG REQUIRED)

hunter_add_package(cub)
find_package(cub CONFIG REQUIRED)

find_package(OpenMP REQUIRED)





set(TRI_CPP_SOURCES
  src/config.cpp
  src/configure.hpp.in
  src/dag_lowertriangular_csr.cpp
  src/dag2019.cpp
  src/logger.cpp
  src/par_graph.cpp
  src/utilities.cpp
  src/reader/gc_tsv_reader.cpp
  src/sparse/unified_memory_csr.cpp
  src/triangle_counter/cpu_triangle_counter.cpp
  src/triangle_counter/nvgraph_triangle_counter.cpp
  src/triangle_counter/triangle_counter.cpp
  src/triangle_counter/cuda_triangle_counter.cpp
)
set(TRI_CPP_HEADERS
    third_party/clara/clara.hpp
    include/graph/config.hpp
    include/graph/dag_lowertriangular_csr.hpp
    include/graph/dag2019.hpp
    include/graph/edge.hpp
    include/graph/edge_list.hpp
    include/graph/logger.hpp
    include/graph/pangolin.hpp
    include/graph/par_graph.hpp
    include/graph/types.hpp
    include/graph/utilities.hpp
    include/graph/allocator/cuda_managed.hpp
    include/graph/allocator/cuda_zero_copy.hpp
    include/graph/dense/cuda_managed_vector.hpp
    include/graph/dense/cuda_zero_copy_vector.hpp
    include/graph/reader/gc_tsv_reader.hpp
    include/graph/triangle_counter/cudamemcpy_tc.hpp
    include/graph/triangle_counter/cpu_triangle_counter.hpp
    include/graph/triangle_counter/cuda_triangle_counter.hpp
    include/graph/triangle_counter/hu_tc.hpp
    include/graph/triangle_counter/impact_2018_tc.hpp
    include/graph/triangle_counter/nvgraph_triangle_counter.hpp
    include/graph/triangle_counter/vertex_tc.hpp
    include/graph/triangle_counter/triangle_counter.hpp
    include/graph/triangle_counter/edge_tc.hpp
)
set(TRI_CUDA_SOURCES
  src/triangle_counter/cudamemcpy_tc.cu
  src/triangle_counter/hu_tc.cu
  src/triangle_counter/impact_2018_tc.cu
  src/triangle_counter/vertex_tc.cu
  src/triangle_counter/edge_tc.cu
)



add_library(pangolin32 SHARED ${TRI_CPP_HEADERS} ${TRI_CUDA_SOURCES} ${TRI_CPP_SOURCES})
add_library(pangolin64 SHARED ${TRI_CPP_HEADERS} ${TRI_CUDA_SOURCES} ${TRI_CPP_SOURCES})
add_library(pangolin::pangolin32 ALIAS pangolin32)
add_library(pangolin::pangolin64 ALIAS pangolin64)



if (CMAKE_BUILD_TYPE MATCHES Debug)
  pangolin_compile_options(
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>
  )
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  pangolin_compile_options(
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
endif()



IF(OPENMP_FOUND)
  tri_compile_definitions(PRIVATE -DUSE_OPENMP)
  tri_link_libraries(PRIVATE OpenMP::OpenMP_C)
  tri_link_libraries(PRIVATE OpenMP::OpenMP_CXX)
  pangolin_compile_options(
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
  )
ENDIF()



pangolin_compile_options(
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--Wno-deprecated-gpu-targets;--expt-extended-lambda;-Xcompiler=-Wall;-Xcompiler=-Wextra>
)


pangolin_compile_options(
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

include(GenerateExportHeader)
generate_export_header(pangolin32)
generate_export_header(pangolin64)

set(include_install_dir "include")
set(lib_install_dir "lib")
set(config_install_dir "${lib_install_dir}/cmake/${PROJECT_NAME}")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

##  Handle the header file with pangolin version info
message(STATUS "${PROJECT_SOURCE_DIR}/src/configure.hpp.in" " -> " "${PROJECT_BINARY_DIR}/include/graph/configure.hpp")
configure_file (
    "${PROJECT_SOURCE_DIR}/src/configure.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/graph/configure.hpp"
)
# install the pangolin version file
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/include/graph/configure.hpp"
  DESTINATION include/graph
)
# include the binary dir during build or the install dir during install
tri_include_directories(PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/graph> # fake that "configure.hpp" is in same dir as pangolin.hpp
  $<INSTALL_INTERFACE:include>
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
    include/graph
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
    TARGETS pangolin32 pangolin64
    EXPORT "${targets_export_name}"
    INCLUDES DESTINATION "${include_install_dir}"
    LIBRARY DESTINATION "${lib_install_dir}"
    ARCHIVE DESTINATION "${lib_install_dir}"
    RUNTIME DESTINATION bin
)

install(
    EXPORT "${targets_export_name}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
)

target_compile_definitions(pangolin64 PRIVATE -DUSE_INT64)

# message(STATUS ${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES})
# tri_include_directories(SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
tri_include_directories(SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

tri_include_directories(PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

tri_include_directories(PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/clara
)

set_tri_properties(PROPERTIES
	# CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


tri_link_libraries(PUBLIC spdlog::spdlog)
tri_link_libraries(PUBLIC cub::cub)
tri_link_libraries(PRIVATE nvgraph)
tri_link_libraries(PRIVATE nvToolsExt)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang or AppleClang
  set_tri_properties(PROPERTIES LINK_FLAGS -Wl,--no-undefined)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  set_tri_properties(PROPERTIES LINK_FLAGS -Wl,--no-undefined)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()

# Request that pangolin and anything requiring it be built with -std=c++11
tri_compile_features(PUBLIC cxx_std_11)
set_tri_properties(PROPERTIES
  CUDA_STANDARD 11
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
)
set_tri_properties(PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
# add the binary dir where the version file is
tri_include_directories(PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>  
)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Setting verbose build during Debug")
  set(CMAKE_VERBOSE_MAKEFILE ON)
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  tri_compile_definitions(PRIVATE -DNDEBUG)
endif()

# Add a target to generate API documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(docs ALL
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
endif(DOXYGEN_FOUND)

