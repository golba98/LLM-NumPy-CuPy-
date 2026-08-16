cmake_minimum_required(VERSION 3.16)
project(codexa_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(codexa_tensor src/tensor.cpp)
target_include_directories(codexa_tensor PUBLIC include)
target_compile_options(codexa_tensor PRIVATE -Wall -Wextra -Wpedantic)

add_executable(tensor_test tests/tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE codexa_tensor)
target_compile_options(tensor_test PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME tensor_test COMMAND tensor_test)
