cmake_minimum_required(VERSION 3.2.0)

project(lstm)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fopenmp -march=native")
set(Eigen3_DIR "/search/odin/imer/packages/eigen-3.3.4/cmake")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

file(GLOB_RECURSE LSTM_CPP "${PROJECT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE MAIN_CPP "${PROJECT_SOURCE_DIR}/main.cpp")
file(GLOB_RECURSE REMOVE_CMAKE "${PROJECT_SOURCE_DIR}/build/*")
list(REMOVE_ITEM LSTM_CPP ${MAIN_CPP})
list(REMOVE_ITEM LSTM_CPP ${REMOVE_CMAKE})
file(GLOB_RECURSE LSTM_H   "${PROJECT_SOURCE_DIR}/*.h")

option (build_Library "Build shared object file" OFF)
option (build_lstm_test "Build test executable file" ON)


if(build_Library)
	add_library(${PROJECT_NAME})
endif()

if(build_lstm_test)
	add_executable("${PROJECT_NAME}_test" ${LSTM_CPP} ${LSTM_H} ${MAIN_CPP})
	target_link_libraries ("${PROJECT_NAME}_test" Eigen3::Eigen)
endif()

