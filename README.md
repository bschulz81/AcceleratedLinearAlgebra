# AcceleratedLinearAlgebra
A library with some linear algebra functions that works with OpenMP and OpenAcc.

This is a first submission for a simple linear algebra library. It is somewhat an extension to the mdspan class of c++.
While mdspan works with compile time set extents, this library uses c++ concepts, so that stl vectors on the heap can be used for extents.
It has support for rowmajor and column mayor data (but that is not suffciently tested) and rank sizes larger than 2 (but that was not tested very much).

Currently, the library uses open-mp and openacc. It contains functions for matrix multiplication on accelerators, as well as advanced and fast algorithms from https://arxiv.org/abs/1812.02056 for Cholesky, LU and QR decomposition (besides the usual vector and matrix calculations). The library also has Strassen's algorithm for matrix multiplication implemented, as well as its Winograd Variant from https://arxiv.org/abs/1410.1599 . The algorithms can be set up such that they revert to naive multiplication on host or on gpu when the matrix size is small enough. And the library can work with data from any object with a pointer, including memory mapped files. By default, Strassen's algorithm as well as the Winograd variant use memmapped files for temporary data.


The Cholesky, LU and QR decomposition can be set such that they work with multiple cores on CPU and use the gpu only for Matrix multiplication, or they can use Strassen's or Winograds's algorithm for the multiplications. However, the algorithms can also work entirely on GPU, in that case, they can only use naive matrix multiplication.

Initial support for the message passing interface was added. But not tested yet. With this, the Strassen algorithm can then send smaller matrices to other nodes, which can be configured such with the MPI that they are on separate computers. Once the matrix is small enough, it will then be uploaded to the gpu, computed, downloaded and send back to the lower rank in the mpi comm world. The remaining parts of the computations are then done with openmp in parallel.

A cmakelists.txt file is supplied. Currently, the library is known to compile on linux with Nvidia's nvc++ compiler from Nvidia's hpc sdk.

Compilation with Gcc currently produces an internal compiler error due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118590 , https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118738 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118794 . For Windows support, one would have to add Windows specific support for memory mapped files. 


The cmakelists.txt creates a project for a simple test application that demonstrates some of the usage of the library with a bit documentation.

Apparently, by default nvc++ assumes that the pointers overlap. As of 04.02, I have used the __restrict and const keyword wherever they seemed to be useful. I also added the option  -Msafeptr=all to the Cmakelists.txt for nvc++. This compile option seems to remove some of the pointer overlapping assumptions of nvc++ and now some optimizations can finally take place. Sometimes nvc++ refuses to vectorize openmp code for reasons it calls "unknown" or for "data dependencies" that do not seem to be there.

Also, apparently, from functions denoted as worker, if they have a sequential loop, one can not call any other parallelized functions, including those denoted as vectors in nvc++ currently. Therefore, the matrix multiplications in the Cholesky/LU/QR decompositions within these loops had to be inlined by copy and paste.  Unfortunately, i have not yet been able to produce gang loop versions of the lu,cholesky and qr decomposition, only worker loop versions. If i convert the procedures into gang routines and turn  the outer loops of the multiplications into gang loops, they would yield different numbers at each run.


On clang, offload problems with open-acc exist and the open_acc version of the library does not work correctly, due to openacc being in the initial stages.


In the development folder, I have now put in an open-mp version of the library. This currently fails with nvc++ because nvc++ does not recognize #pragma omp begin declare target, but instead wants a #pragma omp declare target, which is rather for variables.

However, on clang, the openmp version of the library compiles and works at runtime, but only if all optimizations for the compiler are switched off. 
I have filed a bug for clang because of this https://github.com/llvm/llvm-project/issues/126342 , and if these problems are fixed, the library may then work in full with clang.

In general, OpenMP would be preferable for a scientific library, because in OpenMP one can have nested parallelism, where parallel for loops can call functions which open other parallel for loops. This is essential if one has complex algorithms where one may have complex procedures within a repeated parallelizable loop, like solving eigenvalue problems which contain parallel loops themselves.



