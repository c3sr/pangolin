include(FindPackageHandleStandardArgs)

SET(NUMA_INCLUDE_SEARCH_PATHS
      ${NUMA}
      /usr/include
      $ENV{NUMA}
      $ENV{NUMA_HOME}
      $ENV{NUMA_HOME}/include
)

SET(NUMA_LIBRARY_SEARCH_PATHS
      ${NUMA}
      /usr/lib
      $ENV{NUMA}
      $ENV{NUMA_HOME}
      $ENV{NUMA_HOME}/lib
)

find_path(NUMA_INCLUDE_DIR
  NAMES numa.h
  PATHS ${NUMA_INCLUDE_SEARCH_PATHS}
  DOC "NUMA include directory")

find_library(NUMA_LIBRARY
  NAMES numa
  HINTS ${NUMA_LIBRARY_SEARCH_PATHS}
  DOC "NUMA library")

if (numa_LIBRARY)
    get_filename_component(NUMA_LIBRARY_DIR ${NUMA_LIBRARY} PATH)
endif()

mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARY_DIR NUMA_LIBRARY)


if (NUMA_LIBRARY AND NUMA_INCLUDE_DIR)
  if (NOT TARGET NUMA::NUMA)
    add_library(NUMA::NUMA UNKNOWN IMPORTED)
    set_target_properties(NUMA::NUMA PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}"
      IMPORTED_LOCATION "${NUMA_LIBRARY}"
    )
  else()
    message(WARNING "NUMA::NUMA is already a target")
  endif()

endif()




find_package_handle_standard_args(NUMA 
DEFAULT_MESSAGE
NUMA_INCLUDE_DIR NUMA_LIBRARY
)

# message(STATUS "numaFOUND: " ${numa_FOUND})
# message(STATUS "numa_LIBRARY: " ${numa_LIBRARY})
# message(STATUS "numa_INCLUDE_DIR: " ${numa_INCLUDE_DIR})
# get_property(propval TARGET numa::numa PROPERTY INTERFACE_LINK_LIBRARIES)
# message(STATUS "INTERFACE_LINK_LIBRARIES: " ${propval})