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
  message(STATUS ${PARSED_ARGS_NAME} ${PARSED_ARGS_SRCS})
  add_executable(${PARSED_ARGS_NAME} ${PARSED_ARGS_SRCS})
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