hunter_add_package(Catch)
find_package(Catch2 CONFIG REQUIRED)

function(pangolin_add_test)
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of boolean arguments
    "NAME" # list of names of mono-valued arguments
    "SRCS" # list of names of nuluti-valued arguments
    ${ARGN} # arguments of the function to parse
  )
  add_executable(${PARSED_ARGS_NAME} ${PARSED_ARGS_SRCS})
  set_target_properties(${PARSED_ARGS_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${PARSED_ARGS_NAME} PUBLIC pangolin::pangolin)
  target_link_libraries(${PARSED_ARGS_NAME} PUBLIC Catch2::Catch2)
  # On the NCSA P9 system, CMake won't link C++ targets against CUDA's link directories
  # maybe the header-only target needs to have the CUDA link language 
  target_link_directories(${PARSED_ARGS_NAME} PUBLIC "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
  add_test(NAME ${PARSED_ARGS_NAME} COMMAND ${PARSED_ARGS_NAME} -a)
endfunction()

function(pangolin_add_gpu_test)
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of boolean arguments
    "NAME" # list of names of mono-valued arguments
    "SRCS" # list of names of nuluti-valued arguments
    ${ARGN} # arguments of the function to parse
  )
  pangolin_add_test(${ARGN})
  set_tests_properties(${PARSED_ARGS_NAME} PROPERTIES LABELS "gpu")
endfunction()

function(pangolin_add_mpi_gpu_test)
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of boolean arguments
    "NAME" # list of names of mono-valued arguments
    "SRCS" # list of names of nuluti-valued arguments
    ${ARGN} # arguments of the function to parse
  )
  pangolin_add_test(${ARGN})
  set_tests_properties(${PARSED_ARGS_NAME} PROPERTIES LABELS "gpu;mpi")
endfunction()

pangolin_add_test(
  NAME test_binary_search
  SRCS test_binary_search.cpp
)

pangolin_add_gpu_test(
  NAME test_broadcast
  SRCS test_broadcast.cu
)

pangolin_add_gpu_test(
  NAME test_buffer
  SRCS test_buffer.cu
)

pangolin_add_test(
  NAME test_complete
  SRCS test_complete.cu
)

pangolin_add_gpu_test(
  NAME test_coo
  SRCS test_coo.cu
)

pangolin_add_gpu_test(
  NAME test_count
  SRCS test_count.cu
)

pangolin_add_gpu_test(
  NAME test_csr
  SRCS test_csr.cu
)

pangolin_add_test(
  NAME test_generator_hubspoke
  SRCS test_generator_hubspoke.cu
)

pangolin_add_gpu_test(
  NAME test_load_balance
  SRCS test_load_balance.cu
)

pangolin_add_gpu_test(
  NAME test_load_balance_gpu
  SRCS test_load_balance_gpu.cu
)

pangolin_add_gpu_test(
  NAME test_logger
  SRCS test_logger.cpp test_logger_helper.cpp
)

pangolin_add_gpu_test(
  NAME test_managed_allocator
  SRCS test_managed_allocator.cu
)

pangolin_add_test(
  NAME test_merge_path
  SRCS test_merge_path.cpp
)

pangolin_add_mpi_gpu_test(
  NAME test-mpi
  SRCS test_mpi.cu
)

pangolin_add_gpu_test(
  NAME test_numa
  SRCS test_numa.cu
)

pangolin_add_gpu_test(
  NAME test_tc_anjur_iyer
  SRCS test_tc_anjur_iyer.cu
)


pangolin_add_gpu_test(
  NAME test_tc_edge_binary
  SRCS test_tc_edge_binary.cu
)

pangolin_add_gpu_test(
  NAME test_tc_edge_linear
  SRCS test_tc_edge_linear.cu
)

pangolin_add_gpu_test(
  NAME test_tc_edge_merge
  SRCS test_tc_edge_merge.cu
)

pangolin_add_gpu_test(
  NAME test_tc_vertex_block_binary
  SRCS test_tc_vertex_block_binary.cu
)

pangolin_add_gpu_test(
  NAME test_tc_vertex_blocks_binary
  SRCS test_tc_vertex_blocks_binary.cu
)

pangolin_add_gpu_test(
  NAME test_tc_vertex_blocks_cache_block_binary
  SRCS test_tc_vertex_blocks_cache_block_binary.cu
)

# pangolin_add_gpu_test(
#   NAME test_tc_edge_warp_dyn
#   SRCS test_tc_edge_warp_dyn.cu
# )

pangolin_add_gpu_test(
  NAME test_topology
  SRCS test_topology.cu
)

pangolin_add_gpu_test(
  NAME test_vector
  SRCS test_vector.cu
)

pangolin_add_gpu_test(
  NAME test_zero
  SRCS test_zero.cu
)

pangolin_add_gpu_test(
  NAME test_zero_copy_allocator
  SRCS test_zero_copy_allocator.cu
)

# pangolin_add_test(
#   NAME test_tc_vertex_linear
#   SRCS test_tc_vertex_linear.cu
# )

# pangolin_add_test(
#   NAME test_flow
#   SRCS test_flow.cu
# )

# pangolin_add_test(
#   NAME test_map
#   SRCS test_map.cu
# )