hunter_add_package(Catch)
find_package(Catch2 CONFIG REQUIRED)

add_executable(test1 
  catch.cpp 
  coo.cu
  vector.cu
)
target_link_libraries(test1 PUBLIC pangolin32)
target_link_libraries(test1 PUBLIC Catch2::Catch)

add_test(NAME test1 COMMAND test1 -a)