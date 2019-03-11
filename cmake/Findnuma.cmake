include(FindPackageHandleStandardArgs)

SET(numa_INCLUDE_SEARCH_PATHS
      ${numa}
      /usr/include
      $ENV{numa}
      $ENV{numa_HOME}
      $ENV{numa_HOME}/include
)

SET(numa_LIBRARY_SEARCH_PATHS
      ${numa}
      /usr/lib
      $ENV{numa}
      $ENV{numa_HOME}
      $ENV{numa_HOME}/lib
)

find_path(numa_INCLUDE_DIR
  NAMES numa.h
  PATHS ${numa_INCLUDE_SEARCH_PATHS}
  DOC "NUMA include directory")

find_library(numa_LIBRARY
  NAMES numa
  HINTS ${numa_LIBRARY_SEARCH_PATHS}
  DOC "NUMA library")

if (numa_LIBRARY)
    get_filename_component(numa_LIBRARY_DIR ${numa_LIBRARY} PATH)
endif()

mark_as_advanced(numa_INCLUDE_DIR numa_LIBRARY_DIR numa_LIBRARY)



find_package_handle_standard_args(numa 
DEFAULT_MESSAGE
numa_INCLUDE_DIR numa_LIBRARY
)

if (numa_FOUND)
  add_library(numa SHARED IMPORTED GLOBAL)
  target_link_libraries(numa INTERFACE ${numa_LIBRARY})
  target_include_directories(numa SYSTEM INTERFACE ${numa_INCLUDE_DIR})
  add_library(numa::numa ALIAS numa)
endif()

# message(STATUS "numaFOUND: " ${numa_FOUND})
# message(STATUS "numa_LIBRARY: " ${numa_LIBRARY})
# message(STATUS "numa_INCLUDE_DIR: " ${numa_INCLUDE_DIR})
# get_property(propval TARGET numa::numa PROPERTY INTERFACE_LINK_LIBRARIES)
# message(STATUS "INTERFACE_LINK_LIBRARIES: " ${propval})