function(pangolin_add_test)
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of boolean arguments
    "NAME" # list of names of mono-valued arguments
    "SRCS" # list of names of nuluti-valued arguments
    ${ARGN} # arguments of the function to parse
  )

  # only add tests as targets if we're in the "pangolin" project
  # if we're included in some other project, don't build tests
if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
  add_executable(${PARSED_ARGS_NAME} ${PARSED_ARGS_SRCS})
 else()
   add_executable(${PARSED_ARGS_NAME} EXCLUDE_FROM_ALL ${PARSED_ARGS_SRCS})
 endif()

  set_target_properties(${PARSED_ARGS_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${PARSED_ARGS_NAME} PUBLIC pangolin::pangolin)

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
  if (MPI_FOUND)
  cmake_parse_arguments(
    PARSED_ARGS # prefix of output variables
    "" # list of names of boolean arguments
    "NAME" # list of names of mono-valued arguments
    "SRCS" # list of names of nuluti-valued arguments
    ${ARGN} # arguments of the function to parse
  )
  pangolin_add_test(${ARGN})
  set_tests_properties(${PARSED_ARGS_NAME} PROPERTIES LABELS "gpu;mpi")
  endif()
endfunction()

pangolin_add_test(
  NAME test_main
  SRCS 
  test_main.cu
  test_array_view.cu
  test_search.cu
  test_bmtx_stream.cu
  test_complete.cu
  test_double_buffer.cu
  test_generator_hubspoke.cu
  test_merge_path.cu
  test_numa.cu
  test_single_header.cu
)

pangolin_add_gpu_test(
  NAME test_broadcast
  SRCS test_broadcast.cu
)

pangolin_add_gpu_test(
  NAME test_buffer
  SRCS test_buffer.cu
)

pangolin_add_gpu_test(
  NAME test_count
  SRCS test_count.cu
)

pangolin_add_gpu_test(
  NAME test_csr
  SRCS test_csr.cu
)

pangolin_add_gpu_test(
  NAME test_csr_val
  SRCS test_csr_val.cu
)

pangolin_add_gpu_test(
  NAME test_csr_binned
  SRCS test_csr_binned.cu
)

pangolin_add_gpu_test(
  NAME test_csr_coo
  SRCS test_csr_coo.cu
)

pangolin_add_gpu_test(
  NAME test_device_buffer
  SRCS test_device_buffer.cu
)

pangolin_add_gpu_test(
  NAME test_device_vector
  SRCS test_device_vector.cu
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
  SRCS test_logger.cu test_logger_helper.cu
)

pangolin_add_mpi_gpu_test(
  NAME test-mpi
  SRCS test_mpi.cu
)

pangolin_add_gpu_test(
  NAME test_rc_stream
  SRCS test_rc_stream.cu
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
  NAME test_tc_vertex_bitvector
  SRCS test_tc_vertex_bitvector.cu
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

pangolin_add_gpu_test(
  NAME test_tc_vertex_dyn
  SRCS test_tc_vertex_dyn.cu
)

pangolin_add_gpu_test(
  NAME test_tc_edge_warp_dyn
  SRCS test_tc_edge_warp_dyn.cu
)

pangolin_add_gpu_test(
  NAME test_tc_edge_warp_dysel
  SRCS test_tc_edge_warp_dysel.cu
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
  NAME test_tc_bissonfatica18
  SRCS test_tc_bissonfatica18.cu
)

pangolin_add_gpu_test(
  NAME test_topology
  SRCS test_topology.cu
)

# pangolin_add_test(
#   NAME test_tc_vertex_linear
#   SRCS test_tc_vertex_linear.cu
# )

