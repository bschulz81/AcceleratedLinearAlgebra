 cmake_minimum_required (VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23 )

#Clang compiler flags
#SET (CMAKE_CXX_COMPILER "clang++" CACHE STRING "C++ compiler" FORCE)
#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenacc -O1 -fopenmp   -fopenmp-targets=nvptx64-nvidia-cuda  -Wall")

#GCC compiler flags. currently show an internal compiler error

#SET (CMAKE_CXX_COMPILER "g++" CACHE STRING "C++ compiler" FORCE)
#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenacc -foffload=nvptx-none  -fcf-protection=none -fno-stack-protector -U_FORTIFY_SOURCE -std=c++23  -no-pie")

#Nvidia's nvc++ compiler. Path should be set manually.
set (LANGUAGES "C;CXX")
find_package(OpenACC REQUIRED)
set (CompilerPath "/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin")
set (CMAKE_CXX_COMPILER "nvc++" CACHE STRING "C++ compiler" FORCE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20   -acc=gpu -mp=multicore -Minfo=accel")

project(arraytest VERSION 1.0)

#set project binary folder:
set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/)

#set project source folder:
include_directories(arraytest ${CMAKE_CURRENT_SOURCE_DIR} /opt/cuda/include/ /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/etc/include_acc/)



#add executable target name (if project is built as executable)
add_executable(arraytest ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/main_acc.cpp)
link_directories(arraytest /opt/nvidia/hpc_sdk/Linux_x86_64/25.1/REDIST/compilers/lib/)

if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
    message(STATUS "Configuring for NVIDIA compiler (nvc++)")
    target_compile_options(arraytest PRIVATE -std=c++20 -acc -gpu=cuda12.6 -Minfo=accel)
    target_link_libraries(arraytest PRIVATE rt m c stdc++ mpi )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Configuring for GNU compiler (gcc)")
    target_compile_options(arraytest PRIVATE -fopenacc -fopenmp -foffload=nvptx-none  -fcf-protection=none -fno-stack-protector -U_FORTIFY_SOURCE -std=c++23  -no-pie )
    target_link_libraries(arraytest PRIVATE rt m c stdc++ omp mpi)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Configuring for Clang compiler")
    target_compile_options(arraytest PRIVATE -fopenacc  -fopenmp   -fopenmp-targets=nvptx64-nvidia-cuda  -Wall )
    target_link_libraries(arraytest PRIVATE rt m c stdc++ gomp mpi)
endif()
