cmake_minimum_required(VERSION 3.12 FATAL_ERROR)


include("cmake/HunterGate.cmake")

HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.55.tar.gz"
    SHA1 "da31d97da47cdde05f958786b6729ba9b773bb17"
)

project(placeholder LANGUAGES C CXX CUDA VERSION 0.0.1)

include("${CMAKE_CURRENT_LIST_DIR}/GraphTargets.cmake")