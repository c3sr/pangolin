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
  target_compile_options(${PARSED_ARGS_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
  target_link_libraries(${PARSED_ARGS_NAME} PUBLIC pangolin32)
  target_link_libraries(${PARSED_ARGS_NAME} PUBLIC Catch2::Catch2)
  add_test(NAME ${PARSED_ARGS_NAME} COMMAND ${PARSED_ARGS_NAME} -a)
endfunction()


pangolin_add_test(
  NAME coo
  SRCS coo.cu
)

pangolin_add_test(
  NAME vector
  SRCS vector.cu
)

pangolin_add_test(
  NAME fc
  SRCS fc.cpp
)

pangolin_add_test(
  NAME tc_coo
  SRCS tc_coo.cu
)

pangolin_add_test(
  NAME count
  SRCS count.cu
)