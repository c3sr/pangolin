# https://github.com/hunter-packages/spdlog
# hunter-packages spdlog uses hunter to retrieve fmt
hunter_config(spdlog VERSION 1.2.1-p0 CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(fmt VERSION 5.2.1 CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)
hunter_config(Catch VERSION 2.6.0 CMAKE_ARGS CMAKE_POSITION_INDEPENDENT_CODE=ON)