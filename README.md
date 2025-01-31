# AcceleratedLinearAlgebra
A library with some linear algebra functions that works with OpenMP and OpenAcc.

This is a first submission for a simple linear algebra library. It is somewhat an extension to the mdspan class of c++.
While mdspan works with compile time set extents, this library uses c++ concepts, so that stl vectors on the heap can be used for extents.
It has support for rowmajor and column mayor data (but that is not suffciently tested) and rank sizes larger than 2 (but that was not tested very much).

The library uses open-mp and openacc. It contains functions for matrix multiplication on accelerators, as well as advanced and fast algorithms from https://arxiv.org/abs/1812.02056 for Cholesky, LU and QR decomposition (besides the usual vector and matrix calculations). The library also has Strassen's algorithm for matrix multiplication implemented, as well as its Winograd Variant from https://arxiv.org/abs/1410.1599 . The algorithms can be set up such that they revert to naive multiplication on host or on gpu when the matrix size is small enough. And the library can work with data from any object with a pointer, including memory mapped files. By default, Strassen's algorithm as well as the Winograd variant use memmapped files for temporary data.

The Cholesky, LU and QR decomposition can be set such that they work with multiple cores on CPU and use the gpu only for Matrix multiplication, or they can use Strassen's or Winograds's algorithm for the multiplications. However, the algorithms can also work entirely on GPU, in that case, they can only use naive matrix multiplication.

Initial support for the message passing interface was added. But not tested yet. With this, the Strassen algorithm can then send smaller matrices to other nodes, which can be configured such with the MPI that they are on separate computers. Once the matrix is small enough, it will then be uploaded to the gpu, computed, downloaded and send back to the lower rank in the mpi comm world. The remaining parts of the computations are then done with openmp in parallel.

A cmakelists.txt file is supplied. Currently, the library is known to compile on linux with Nvidia's nvc++ compiler from Nvidia's hpc sdk.

Compilation with Gcc currently produces an internal compiler error: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118590 . For Windows support, one would have to add Windows specific support for memory mapped files. 

On clang, offload problems exist and the library does not work correctly, due to openacc being in the initial stages.

The cmakelists.txt creates a project for a simple test application that demonstrates some of the usage of the library with a bit documentation.



.

