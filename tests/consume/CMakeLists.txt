cmake_minimum_required(VERSION 3.28)

# Standalone smoke project. It is deliberately NOT added to the parent build:
# the point is to prove that an install prefix is consumable from outside the
# source tree, through find_package alone.
#
#   cmake --install <build> --prefix /tmp/ckl-prefix
#   cmake -S tests/consume -B /tmp/ckl-consume -DCMAKE_PREFIX_PATH=/tmp/ckl-prefix
#   cmake --build /tmp/ckl-consume && /tmp/ckl-consume/consume

project(ckl_consume LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CKL REQUIRED)

add_executable(consume main.cpp)
target_link_libraries(consume PRIVATE ckl::ckl)
