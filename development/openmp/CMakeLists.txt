 cmake_minimum_required (VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23 )

project(arraytest VERSION 1.1.1)


set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/)
set (LANGUAGES "C;CXX")

#GCC compiler flags.
#SET (CMAKE_CXX_COMPILER "/usr/bin/g++-16" CACHE STRING "C++ compiler" FORCE)
#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -foffload=nvptx-none -fno-stack-protector -fno-pie  -Wall")

#mpicc compiler flags.
#SET (CMAKE_CXX_COMPILER "mpic++" CACHE STRING "C++ compiler" FORCE)
#SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -foffload=nvptx-none -fno-stack-protector  -Wall")


#Clang compiler flags
SET (ENV{LIBRARY_PATH} "/usr/lib64/nvptx64-nvidia-cuda/:$ENV{LIBRARY_PATH}")
SET (CMAKE_CXX_COMPILER /usr/lib/llvm/21/bin/clang++  CACHE STRING "C++ compiler" FORCE)
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++23 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Wall")


#Nvidia's nvc++ compiler. Path should be set manually.

#set (CMAKE_CXX_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nvc++" CACHE STRING "C++ compiler" FORCE )
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23  -mp=gpu -Minfo=all")



#arraytest is an executable that tests functions with openmp and gpu offload

#arraytest_mpi is an executable that tests functions with the message passing interface.
#it is to be run with  mpirun -np 8 ./arraytest_mpi


include_directories(datablocktests      ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(mdspantests         ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(mathdemonstrations  ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(arraytest_mpi       ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(sparsetests         ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(datablocktests           ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/datablocktests.cpp)
add_executable(mdspantests              ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/mdspantests.cpp)
add_executable(mathdemonstrations       ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/mathdemonstrations.cpp)
add_executable(arraytest_mpi            ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/main_mpi.cpp)
add_executable(sparsetests              ${SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/sparsetests.cpp)

#Set this to if you want to use unified_shared_memory support
#add_compile_definitions(Unified_Shared_Memory)




if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
    link_directories(datablocktests             /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/REDIST/compilers/lib/)
    link_directories(mdspantests                /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/REDIST/compilers/lib/)
    link_directories(mathdemonstrations         /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/REDIST/compilers/lib/)
    link_directories(arraytest_mpi              /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/REDIST/compilers/lib/)
    link_directories(sparsetests                /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/REDIST/compilers/lib/)
    target_link_libraries(datablocktests        PRIVATE         rt m c stdc++ mpi  )
    target_link_libraries(mdspantests           PRIVATE         rt m c stdc++ mpi )
    target_link_libraries(mathdemonstrations    PRIVATE         rt m c stdc++ mpi  )
    target_link_libraries(arraytest_mpi         PRIVATE         rt m c stdc++ mpi  )
    target_link_libraries(sparsetests           PRIVATE         rt m c stdc++ mpi  )
else()
    target_link_libraries(datablocktests        PRIVATE         rt m c stdc++ mpi  )
    target_link_libraries(mdspantests           PRIVATE         rt m c stdc++ mpi  )
    target_link_libraries(mathdemonstrations    PRIVATE         rt m c stdc++ mpi )
    target_link_libraries(arraytest_mpi         PRIVATE         rt m c stdc++ mpi  )
    target_link_libraries(sparsetests           PRIVATE         rt m c stdc++ mpi  )
endif()


