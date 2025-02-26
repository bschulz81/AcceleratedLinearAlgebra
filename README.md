# AcceleratedLinearAlgebra
A library with some linear algebra functions that works with OpenMP and Open-MPI on the host and on gpu accelerators.

This is a first submission for a simple linear algebra library. It is somewhat an extension to the mdspan class of c++.
While mdspan works with compile time set extents, this library uses c++ concepts, so that stl vectors on the heap can be used for extents.
It has support for rowmajor and column mayor data (but that is not suffciently tested) and rank sizes larger than 2 (but that was not tested very much).

Currently, the library uses open-mp. An older version that contains openmp code for the host and open-acc code for the gpu offload is in the archive folder.

The library contains functions for matrix multiplication on accelerators, as well as advanced and fast algorithms from https://arxiv.org/abs/1812.02056 for Cholesky, LU and QR decomposition (besides the usual vector and matrix calculations). The library also has Strassen's algorithm for matrix multiplication implemented, as well as its Winograd Variant from https://arxiv.org/abs/1410.1599 . The algorithms can be set up such that they revert to naive multiplication on host or on gpu when the matrix size is small enough. And the library can work with data from any object with a pointer, including memory mapped files. By default, Strassen's algorithm as well as the Winograd variant use memmapped files for temporary data.


The Cholesky, LU and QR decomposition can be set such that they work with multiple cores on CPU and use the gpu only for Matrix multiplication, or they can use Strassen's or Winograds's algorithm for the multiplications. However, the algorithms can also work entirely on GPU, in that case, they can only use naive matrix multiplication.

Initial support for the message passing interface was added. But not tested yet. With this, the Strassen algorithm can then send smaller matrices to other nodes, which can be configured such with the MPI that they are on separate computers. Once the matrix is small enough, it will then be uploaded to the gpu, computed, downloaded and send back to the lower rank in the mpi comm world. The remaining parts of the computations are then done with openmp in parallel.

A cmakelists.txt file is supplied. Currently, the library is known to compile on linux with clang. Since 17.02.2025, it runs and compiles with -O3 optimizations switched on.

Compilation with Gcc currently produces an internal compiler error due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118590 , https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118738 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118794 . For Windows support, one would have to add Windows specific support for memory mapped files. 


On 13.02.2025, I fixed a data race which led in some cases to a wrong calculation of the offloaded datalength. now the openmp code yields the same results as the old openacc code.
After running them a few hundreds of times and with different matrices, I can by 13.02.2025, asses that they appear to work correctly if the gpu driver and the cuda version are the most recent.

On 17.02.2025, The algorithms for the gpu where rewritten such that they now use teams of threads as often as possible. 

Also, initial support for shared memory was added, but I was unable to test it, since my gpu has shared memory but is too old that clang would be able to use it. Due to Openmp's restrictions on the teams distribute pragma, the use of teams of threads is in some cases only possible with shared memory. On some cases with reductions, one unfortunately still has to use threads with parallel for. 
