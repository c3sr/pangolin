
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


project(tri LANGUAGES C CXX CUDA VERSION 0.1.0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})



hunter_add_package(spdlog)
find_package(spdlog CONFIG REQUIRED)

find_package(OpenMP REQUIRED)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()


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
)

message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
			   " archs: ${CUDA_ARCH_FLAGS_readable})")

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_VERBOSE_BUILD OFF)

set(TRI_CPP_SOURCES
    src/logger.cpp
    src/main.cpp
    src/triangle_counter.cpp
)
set(TRI_CPP_HEADERS
    third_party/clara/clara.hpp
    include/graph/logger.hpp
)
set(TRI_CUDA_SOURCES
    src/utilities.cu
    src/gpu_triangle_counter.cu
)


add_executable( tri ${TRI_CPP_HEADERS} ${TRI_CUDA_SOURCES} ${TRI_CPP_SOURCES})

if (CUDA_FOUND)
  target_include_directories(tri PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

# Add include directories
target_include_directories(tri PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/third_party/clara
)

set_target_properties(tri PROPERTIES
	# CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


# target_link_libraries(scope fmt::fmt)
target_link_libraries(tri spdlog::spdlog)
target_link_libraries(tri ${CUDA_LIBRARIES})
target_link_libraries(tri ${OpenMP_LIBRARIES})
# target_link_libraries(scope nvToolsExt)

# Request that scope be built with -std=c++11
# As this is a public compile feature anything that links to
# scope will also build with -std=c++11
target_compile_features(tri PUBLIC cxx_std_11)



# Convert CUDA flags from list
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

