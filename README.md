# AcceleratedLinearAlgebra
A library with some linear algebra functions that works with OpenMP and OpenAcc.

This is a first submission for a simple linear algebra library. It is somewhat an extension to the mdspan class of c++.
While mdspan works with compile time set extents, this library uses c++ concepts, so that stl vectors on the heap can be used for extents.
It has support for rowmajor and column mayor data (but that is not suffciently tested) and rank sizes larger than 2 (but that was not tested very much).

The library uses open-mp and openacc. It contains functions for matrix multiplication on accelerators, as well as advanced and fast algorithms from https://arxiv.org/abs/1812.02056 for Cholesky, LU and QR decomposition (besides the usual vector and matrix calculations). The library also has Strassen's algorithm for matrix multiplication implemented, as well as its Winograd Variant from https://arxiv.org/abs/1410.1599 . The algorithms can be set up such that they revert to naive multiplication on host or on gpu when the matrix size is small enough. And the library can work with data from any object with a pointer, including memory mapped files. By default, Strassen's algorithm as well as the Winograd variant use memmapped files for temporary data.


The Cholesky, LU and QR decomposition can be set such that they work with multiple cores on CPU and use the gpu only for Matrix multiplication, or they can use Strassen's or Winograds's algorithm for the multiplications. However, the algorithms can also work entirely on GPU, in that case, they can only use naive matrix multiplication.

Initial support for the message passing interface was added. But not tested yet. With this, the Strassen algorithm can then send smaller matrices to other nodes, which can be configured such with the MPI that they are on separate computers. Once the matrix is small enough, it will then be uploaded to the gpu, computed, downloaded and send back to the lower rank in the mpi comm world. The remaining parts of the computations are then done with openmp in parallel.

A cmakelists.txt file is supplied. Currently, the library is known to compile on linux with Nvidia's nvc++ compiler from Nvidia's hpc sdk.

Compilation with Gcc currently produces an internal compiler error due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118590 , https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118738 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518. For Windows support, one would have to add Windows specific support for memory mapped files. 

On clang, offload problems exist and the library does not work correctly, due to openacc being in the initial stages.

The cmakelists.txt creates a project for a simple test application that demonstrates some of the usage of the library with a bit documentation.

As of 04.02, I have used the __restrict and const keyword wherever they seemed to be useful. Apparently, by default nvc++ assumes that the pointers overlap (which I find quite strange, since this implies that the assumption is that a programmer writes a vector loop without knowing when this is possible. If given the restrict keyword, the gpu code is now much faster. Most warnings of non-parallelizable code are issued for functions and loops where no parallelization should happen (e.g. a printmatrix function). But there are still some issues with openmp loops. Openmp has no "independent" clause as it assumes the programmer to know what he does. Sometimes nvc++ refuses to vectorize openmp code for reasons it calls "unknown" or for "data dependencies" that are clearly not there. The matrix multiplication can be parallelized with openacc on the device but nvc++ refuses to parallelize the exactly same code with open-mp pragmas for the host.

On 04.02.25, I now added the option  -Msafeptr=all to the Cmakelists.txt for nvc++. This seems to remove some of the pointer overlapping assumptions of nvc++ and now some optimizations can finally take place, but I have to go through the entire report to see the results of this option.
 

Also, apparently, from functions denoted as worker, if they have a sequential loop, one can not call any other parallelized functions, including those denoted as vectors in nvc++ currently. Therefore, the matrix multiplications in the Cholesky/LU/QR decompositions within these loops had to be inlined. Also, vector loops in the constructors of the datastruct class were removed in order to prevent crashes. In the current applications, these loops would not even be called, but for tensors, this could lead to a slower creation, since some of the loops in these constructors for tensors with rank>2 could be parallelized in theory.
