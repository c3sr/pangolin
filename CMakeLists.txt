
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include("cmake/HunterGate.cmake")

HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.55.tar.gz"
    SHA1 "da31d97da47cdde05f958786b6729ba9b773bb17"
)

option(CONFIG_USE_HUNTER "Turn on to enable using the hunteger package manager" ON)
option(CUDA_MULTI_ARCH "Whether to generate CUDA code for multiple architectures" OFF)
option(USE_OPENMP "compile with OpenMP support" ON)
option(USE_CUDA "compile with CUDA support" ON)
option(USE_NVGRAPH "build with NVGraph support" ON)


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

find_package(OpenMP REQUIRED)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS})
ENDIF()


if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -G)
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
  src/dag2019.cpp
  src/edge_list.cpp
  src/logger.cpp
  src/main.cpp
  src/cpu_triangle_counter.cpp
  src/nvgraph_triangle_counter.cpp
  src/triangle_counter.cpp
  src/utilities.cpp
)
set(TRI_CPP_HEADERS
    third_party/clara/clara.hpp
    include/graph/cpu_triangle_counter.hpp
    include/graph/cudamemcpy_tc.hpp
    include/graph/dag2019.hpp
    include/graph/edge_list.hpp
    include/graph/gpu_triangle_counter.hpp
    include/graph/logger.hpp
    include/graph/nvgraph_triangle_counter.hpp
    include/graph/triangle_counter.hpp
    include/graph/utilities.hpp
)
set(TRI_CUDA_SOURCES
    src/gpu_triangle_counter.cu
    src/cudamemcpy_tc.cu
)


add_executable(tri32 ${TRI_CPP_HEADERS} ${TRI_CUDA_SOURCES} ${TRI_CPP_SOURCES})
add_executable(tri64 ${TRI_CPP_HEADERS} ${TRI_CUDA_SOURCES} ${TRI_CPP_SOURCES})

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


# target_link_libraries(scope fmt::fmt)
tri_link_libraries(spdlog::spdlog)
tri_link_libraries(${CUDA_LIBRARIES})
tri_link_libraries(${OpenMP_LIBRARIES})
# target_link_libraries(scope nvToolsExt)

# Request that scope be built with -std=c++11
# As this is a public compile feature anything that links to
# scope will also build with -std=c++11
tri_compile_features(PUBLIC cxx_std_11)



# Convert CUDA flags from list
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

