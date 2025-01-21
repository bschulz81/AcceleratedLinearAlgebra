cmake_minimum_required (VERSION 3.10)

set(CMAKE_CXX_STANDARD 23 )

#Clang compiler flags
#SET (CMAKE_CXX_COMPILER "clang++" CACHE STRING "C++ compiler" FORCE)
#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenacc -O1 -fopenmp -g  -fopenmp-targets=nvptx64-nvidia-cuda  -Wall")

#GCC compiler flags. currently show an internal compiler error

#SET (CMAKE_CXX_COMPILER "g++" CACHE STRING "C++ compiler" FORCE)
#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenacc -foffload=nvptx-none  -fcf-protection=none -fno-stack-protector -U_FORTIFY_SOURCE -std=c++23  -no-pie")

#Nvidia's nvc++ compiler. Path should be set manually.

SET (CompilerPath "/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin")
SET (CMAKE_CXX_COMPILER "${CompilerPath}/nvc++" CACHE STRING "C++ compiler" FORCE)
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -acc -mp=multicore  -Minfo=all")

project(arraytest VERSION 1.0)

#set project binary folder:
SET(PROJECT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/)

#set project source folder:
include_directories(arraytest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


#add executable target name (if project is built as executable)
add_executable(arraytest ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/main_acc.cpp)

target_link_libraries(arraytest rt m c stdc++)
