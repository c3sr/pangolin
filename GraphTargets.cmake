

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include("cmake/GetGitRevisionDescription.cmake")


get_git_head_revision(GIT_REFSPEC GIT_HASH)
git_local_changes(GIT_LOCAL_CHANGES)
message(STATUS GIT_REFSPEC=${GIT_REFSPEC})
message(STATUS GIT_HASH=${GIT_HASH})
message(STATUS GIT_LOCAL_CHANGES=${GIT_LOCAL_CHANGES})

option(CONFIG_USE_HUNTER "Turn on to enable using the hunteger package manager" ON)
option(CUDA_MULTI_ARCH "Whether to generate CUDA code for multiple architectures" OFF)
option(USE_OPENMP "compile with OpenMP support" ON)
option(USE_CUDA "compile with CUDA support" ON)
option(USE_NVGRAPH "build with NVGraph support" ON)

set(
    CMAKE_TOOLCHAIN_FILE
    "${CMAKE_CURRENT_LIST_DIR}/cmake/toolchain.cmake"
    CACHE
    FILEPATH
    "Default toolchain"
)

project(tri LANGUAGES C CXX CUDA VERSION 0.1.0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

macro(tri_include_directories)
  target_include_directories(tri32 ${ARGN})
  target_include_directories(tri64 ${ARGN})
endmacro()
macro(tri_link_libraries)
  target_link_libraries(tri32 ${ARGN})
  target_link_libraries(tri64 ${ARGN})
endmacro()
macro(set_tri_properties)
  set_target_properties(tri32 ${ARGN})
  set_target_properties(tri64 ${ARGN})
endmacro()
macro(tri_compile_features)
  target_compile_features(tri32 ${ARGN})
  target_compile_features(tri64 ${ARGN})
endmacro()
macro(tri_compile_definitions)
  target_compile_definitions(tri32 ${ARGN})
  target_compile_definitions(tri64 ${ARGN})
endmacro()

hunter_add_package(spdlog)
find_package(spdlog CONFIG REQUIRED)

hunter_add_package(cub)
find_package(cub CONFIG REQUIRED)

find_package(OpenMP REQUIRED)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS})
ENDIF()




find_package(CUDA REQUIRED)
if (CUDA_FOUND)
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -G)
  elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -lineinfo)
  endif()
endif()

# Disable extended variants of C++ dialects
# i.e. don't choose gnu++17 over c++17
set(CMAKE_CXX_EXTENSIONS OFF)


# CUDA flags
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMake FindCUDA auto seems to add unsupported architectures somtimes, so we allow the user
# to override with NVCC_ARCH_FLAGS
if(CUDA_MULTI_ARCH)
  CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS All)
else()
  if (DEFINED NVCC_ARCH_FLAGS)
    message(STATUS "Manual cuda arch flags...")
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${NVCC_ARCH_FLAGS})
  else()
    message(STATUS "Automatic cuda arch flags...")
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)
  endif()
endif()

# FIXME: we should not modify these globally
LIST(APPEND CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS}
					         -Wno-deprecated-gpu-targets
					         --expt-extended-lambda
                   -Xcompiler -Wall
                   -Xcompiler -Wextra
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
			   " archs: ${CUDA_ARCH_FLAGS_readable})")

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_VERBOSE_BUILD OFF)

set(TRI_CPP_SOURCES
  src/config.cpp
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
    include/graph/configure.hpp.in
    include/graph/dag_lowertriangular_csr.hpp
    include/graph/dag2019.hpp
    include/graph/edge.hpp
    include/graph/edge_list.hpp
    include/graph/logger.hpp
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

install(
  FILES
    ${TRI_CPP_HEADERS}
  DESTINATION
    include
  COMPONENT
    Devel
)

add_library(tri32 SHARED ${TRI_CPP_HEADERS} ${TRI_CUDA_SOURCES} ${TRI_CPP_SOURCES})
add_library(tri64 SHARED ${TRI_CPP_HEADERS} ${TRI_CUDA_SOURCES} ${TRI_CPP_SOURCES})
include(GenerateExportHeader)
generate_export_header(tri32)
generate_export_header(tri64)

install(TARGETS tri32 tri64 EXPORT graphTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/graph/graphConfigVersion.cmake"
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(
  FILES
    cmake/toolchain.cmake
    GraphConfig.cmake
    GraphTargets.cmake
  DESTINATION
    lib/cmake/graph
)



target_compile_definitions(tri64 PRIVATE -DUSE_INT64)

if (OPENMP_FOUND)
  tri_compile_definitions(PRIVATE -DUSE_OPENMP)
endif()


if (CUDA_FOUND)
  tri_include_directories(SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

tri_include_directories(PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/third_party/clara
)

set_tri_properties(PROPERTIES
	# CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


# FIXME: these should be public
tri_link_libraries(spdlog::spdlog)
tri_link_libraries(${OpenMP_LIBRARIES})
if(CUDA_FOUND)
  tri_link_libraries(${CUDA_LIBRARIES} -lnvgraph)
  tri_link_libraries(cub::cub)
  tri_link_libraries(nvToolsExt)
endif()

# Request that scope be built with -std=c++11
# As this is a public compile feature anything that links to
# scope will also build with -std=c++11
tri_compile_features(PUBLIC cxx_std_11)

# Generate version file and include it 
configure_file (
    "${PROJECT_SOURCE_DIR}/include/graph/configure.hpp.in"
    "${PROJECT_BINARY_DIR}/include/graph/configure.hpp"
)
tri_include_directories(PRIVATE
  ${PROJECT_BINARY_DIR}/include
)

# FIXME: this should not be modified
# Convert CUDA flags from list
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_VERBOSE_MAKEFILE ON)
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  tri_compile_definitions(PUBLIC -DNDEBUG)
endif()

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

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


include(GenerateExportHeader)