# AcceleratedLinearAlgebra
A library with some linear algebra functions that works with OpenMP and Open-MPI on the host and on gpu accelerators.

This is a first submission for a simple linear algebra library. It is somewhat an extension to the mdspan class of c++.
While mdspan works with compile time set extents, this library uses c++ concepts, so that stl vectors on the heap can be used for extents.
It has support for rowmajor and column mayor data and rank sizes larger than 2 (but higher ranks are currently slow and not parallelized).

Currently, the library uses open-mp on cpu and gpu and the message passing interface.

An older version that contains openmp code for the host and open-acc code for the gpu offload is in the archive folder.

The library contains functions for matrix multiplication on accelerators, as well as advanced and fast algorithms from https://arxiv.org/abs/1812.02056 for Cholesky, LU and QR decomposition (besides the usual vector and matrix calculations). The library also has Strassen's algorithm for matrix multiplication implemented, as well as its Winograd Variant from https://arxiv.org/abs/1410.1599 . The algorithms can be set up such that they revert to naive multiplication on host or on gpu when the matrix size is small enough. And the library can work with data from any object with a pointer, including memory mapped files. 

For simple problems, the Cholesky, LU and QR decomposition can be set such that they work with multiple cores on CPU, or on GPU. These variants will then use all three parallelization levels of the GPU. 
For problems that are larger than the memory of the GPU, advanced algorithms are available, which can use the Strassen algorithm or its Winograd variant, to separate the problem into smaller sub-problems, which can then 
be computed on gpu. The algorithms can be configured when to offload, or they choose this automatically.

Functions that involve large sums, like matrix multiplication, or matrix vector multiplication or vector dot product have variants with Kahan summation, but I have not tested this feature much.
Initial support for sparse matrices and sparse times sparse to dense matrix multiplication was added. In order to support a maximum amount of parallelization, I store the matrices dense and add indices of non-zero blocks. The multiplication then just visits overlapping non-zero blocks of the matrices.

The provided cmakelists.txt compiles two test applications. One demonstrates the gpu offload and the parameters for various algorithms.
The other demonstrates the message passing interface use of the library and can be run with 

mpirun -np 12 ./arraytest_mpi 

Be sure to use more or equal nodes than are needed by the recursion. The Strassen algorithm and its Winograd variant, as well as the advanced algorithms for Cholesky, Lu, and Qr decomposition can now auto-configure itself with the help of policy classes to decide whether they should work on gpu or adapt themselves for the message passing interface and gpu offload. 



The library currently compiles with clang 21.1.2 and produces the correct output for this compiler.

A short tutorial how to configure clang and gcc for gpu-offload is here for the gentoo linux distribution: https://forums.gentoo.org/viewtopic-p-8848457.html?sid=7f023fe73bf270b0617ea00bcc1a4ea1


# Todo:

0) Test the numerical stability and correct compilation of the algorithms even more with larger data. 
1) Test the new expanded support for the message passing interface support more intensely and refine them for usage.
2) Add options for the linear algebra functions such that most or all of them can use the message passing interface as well as the gpu then for local work.
3) add functions for statistics, function minimization, auto differentiation, optimization, differential equations

# Version history
### 15.10.2025

Currently, the library does not work with gcc due to the compiler bugs https://gcc.gnu.org/bugzilla/show_bug.cgi?id=122281 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=122280 which I now filed for their openmp nvptx target.

The clang compiler in version 21.1.3 compiles the code correctly. 

Unlike gcc, clang currently has no support for the target simd directive on nvptx targets. I filed a feature request for this as the library uses this extensively https://github.com/llvm/llvm-project/issues/163335

### 08.10.2025:

Optimizations for the Strassen and Winograd Algorithm. 
Added more support for operations with sparse matrices but I have not tested these sparsse operations yet.

After the developers of the clang compiler fixed some bugs in version 21.1.2, the library now compiles and runs with clang 21.1.2

Fortunately, after the last fixes, https://releases.llvm.org/21.1.0/tools/clang/docs/ReleaseNotes.html#openmp-support the library, in the version from 08.10.2025, compiles and runs with clang 21.1.2, kernel 6.17.1 and nvidia-drivers nvidia-drivers-580.95.05. It also produces the correct output when compiled with clang. 


### 28.09.2025:

Added  initial support for sparse matrices and sparse matrix multiplication on cpu and gpu. 
Fixed a typo in matrix vector multiply. 
Matrix Vector multiply now has variants with Kahan summation.


### 23.09.2025:

Renamed the datastruct class into DataBlock, in order to have a consistent naming scheme when I add support for fast handling of sparse tensors with a class holding an array of DataBlock classes.
Fixed a missing requires requires declaration of the assign method in DataBlock
Fixed a test for the math functions. The Multiply should be made with B.transpose as otherwise the result would not make sense. Corrently, the mathematical functions do not check the correct dimensions of the matrices in order be as fast as possible. Perhaps I should add such tests and return error values.



### 19.09.2025:

Separated the various demonstrations how the classes work,and the tests of the classes into several programs. 
Added many test cases and demonstrations of the library.  
Added a nice syntax for expression manipulation. 
Now, one can write expressions like C=A*B to  multiply matrices and matrices and vectors or matrices and scalars, or C=A+B, C=A-B to add or subtract matrices or vectors. These expresisions use a policy class by default, which decides whether to offload to GPU, which is a bit slow.
Implementing the ability to write chains like D=A*(B+C..) would be a mere line. The problem is it would require the storage of intermediate objects. 
While the library provides a class which can do this with ease, the problem is that if, say B in D=A*B+C is 1 GB large, the library would then automatically have to create large intermediate matrices on gpu or cpu or harddrive, and the user would give up the control to order every allocation explicitely. I am still thinking whether to implement something like this.
Added some speed optimizations for the library with const arguments.
Added constructors which make the declaration of vectors in datastruct and mdspan easier,
Changed the order of arguments of the constructors in mdspan to make it more usable for humans.
Changed the extraction of rows, columns and submatrices such that they collapse the rank automatically.


### 17.09.25:

Updated the main_mpi.cpp file to use the new printtensor() function instead of the removed printmatrix function, 
The printtensor function prints tensors residing on host as well as on device and can work with tensors of all ranks.

### 16.09.25:

Speed improvements in the datastruct class for the supspanmatrix, subspan, row and column extraction methods, and the matrix multiplication.
The column and row methods have now rank reducing and rank preserving forms
The printtensor method now works with device data too
The mdspan class can now work with device pointers.
The mdspan class has a mapping manager that is shared among instances and provides book keeping to prevent overlapping mappings, which are forbidden in the openmp standard.
The mdspan data class can now create data purely on device. 
The code of the mdspan and mdspan_data and datastruct classes was significantly polished, improved and tested for various parameters and circumstances.
The test application demonstrates more basic matrix and tensor access on device and on host.

### 11.09.25:

Added test cases for simple matrix, vector and tensor operations with row and column major data.
Reworked the datastruct class to ensure the matrix and tensor functions work for higher than rank 2,
updated the mdspan class, which does not own the data, but strides and extents, to clean up the memory consistently.

Next is a rework of the mdspan_data class. Then, I will revisit the message passing interface functions and test them extensively and add matrix and tensor functions where necessary to full mpi support. 
After that one can then finally do numerical mathematics and add function optimizers and support for differential equations and statistics.



### 09.09.25:
Fixed the class in datastruct.h to accomodate for column major matrices in addition to the rowmajor case which was used in the algorithms earlier. I added test cases for column major data

### 06.09.25:
Fixed a bug in the gpu version of the advancet algorithms for the Cholesky decomposition. All the Algorithms from https://arxiv.org/pdf/1812.02056 as well as the Strassen Algorithm and Its Winograd variant https://arxiv.org/abs/1410.1599 now work on device.

### 06.09.25,

0) The test applications now verify the correctness of the algorithms with matrices that are a bit larger than before.

1) The advanced algorithms for LU/ Cholesky and QR decomposition as well as the Strassen and Winograd algorithms can now work if they are given gpu data pointers.
2) However, extents and strides should be host pointers, because, due to speed, the algorithms create submatrices, which just need changed offsets, strides and extents (2 element arrays) that can be easily offloaded, on the host.


Unfortunately, the advanced algorithm for the QR decomposition  from https://arxiv.org/pdf/1812.02056 showed severe numerical stability errors, which are inherent in the
mathematics of the algorithms from the paper. I have included some measures to increase numerical stability. The instability arises because the advanced algorithms use
the Strassen algorithm twice for one matrix multiplication after another and then a Grahm Schmidt orthonormalization procedure.
The Strassen algorithm replaces multiplications by faster additions, which are, however, numerically unstable.

The Grahm Schmidt, and any other (even an improved ) orthonormalization procedure uses dot products that involve large sums over columns of matrices. These are also numerically unstable.
So the algorithm from https://arxiv.org/pdf/1812.02056  employs three numerically unstable methods in a chain, where the instability increases with the size of the matrices.

For my test data, I found that it could be stabilized a bit by replacing the first Strassen multiplication
by an ordinary one. So I now have set this to the ordinary multiplication in general. For the other multiplications, the option to use the naive multiplication instead of the Strassen can be set by the user if he 
wants to improve stability. In fact, the naive method is used by the policy as default.

However, given that the error of any dot product between columns of a matrix increases with matrix size, I need to test stability even more for larger data.
In order to increase precision, I have began to add methods for Kahan sums for products.

Of course the library is also able to use the simple algorithms for the QR decomposition on gpu, which is not affected by stability problems from Strassen multiplication, but any QR decomposition
(even those with improved Grahm Schmidt orthogonalization) need dot products of vectors and is affected by numerical instability of large sums. 


### 03.09.2025:
 
 A bug in the memory functions for gpu was fixed that confused offloading in recursive and repeated cases.
 As a consequence, issues in the gpu versions of the Strassen algorithm and its Winograd version are resolved.
 The Strassen and Winograd algorithms can now work on device, with devicepointers for the data supplied. 
 The Winograd version of the Strassen algorithm was optimized for more performance and less memory consumption.



### 31.08.2025:

I fixed a bug in the matrix*vector multiply function. 

I rewrote the library into several classes in different files, which are easily testable. 

Now the project consists of one basic datastruct class, on which mathematical function can operate and which can be offloaded to gpu, one mdspan child class which can host strides and extents, 
another childclass called mdspan_data, which hosts the data as well with a shared pointer. Additionally a policy model for the mathematical algorithms was added. This policy model is able to autoconfigure options and 
can decide automatically, whether the function should offload to gpu, or whether the message passing interface is used in the Strassen algorithm. This makes the library slower than numpy, which does not need to check whether the data can be offloaded to a device or whether the message passing interface comm world size is large enough. 

The mathematical functions were separated into classes for gpu and openmp, such that they could be easily replaced by Fortran functions for improved speed.

The Strassen algorithm, as well as the more advanced algorithms for Cholesky, Lu and Qr decomposition now are able to work entirely on GPU, but unfortunately only in unified shared memory mode, due to the recursive nature.
I was not able to run complex algorithms with recursion, subdata and temporary data in offload kernels when compiling with gcc. It always would then claim that there are illegal accesses, or synchronization problems.
This may have been a problem with the gpu or the driver. 

For GPUs which do not have unified shared memory, simpler algorithms for Cholesky, LU and Qr decomposition that the library offers, work on GPU, using all parallelization levels.

I Added more message passing interface support. But most of that is untested until now.



### 12.08.2025:

In addition to the linear algebra functions for mdspan, some blas functions for execution within a gpu kernel additionally needed to be fixed for the different operators in which the strides do not occur as arguments. 
I overlooked this on 11.08, since there was no warning from gcc, as some of functions were not called in the test application. This merely does an LU/Cholesky/QR decomposition on gpu and a strassen algorithm, and for this
it offloads data of an mdspan class to gpu and computes, but does not! call other blas routines of the library within a kernel. I suspect I have to add more test cases.....

I fixed them now and prepared the header a bit for refractoring tomorrow.
A refractoring with inheritance and several classes in different files will give the library a clearer structure, which is also needed for thorough separate testing of the components. 




### 11.08.2025:

A severe bug was discovered in the Strassen and Winograd algorithms. In order to improve optimization I had added the strides to the () operators of the tensors. This caused difficulties with computations over matrices. I accidentially used two indices, instead of four in a computation of the aforementioned algorithms. This caused wrong results. I now changed this such that the () operators do not need strides. The algorithms now work correctly. I also tested them, in addition to OpenMP, with the Message Passing Interface.

The Strassen and Winograd algorithms now work correctly with the Message Passing interface. 

They can distribute the problem on many nodes, and then, if it is small enough, upload on gpu.
So ideally, one sets up one node per unit consisting of a processor with a gpu. More algorithms for the message passing interface may be added in the future.


### 07.08.2025: 

Some OpenMP shared clauses were fixed,
MPI recieve was put as a constructor into the mdspan class,
MPI send was put in as a method, for the entire class with span fields, and for just the data.
Some Message Passing Interface functions (MPI Send, MPI recieve, MPI Bcast were tested. The test application was updated and now compiles a second application with the OpenMPI replacement compiler.

It can be run with  mpirun -np 12 ./arraytest_mpi 

Unfortunately, the Strassen Algorithm and its Winograd version still have problems and crash when using the Message Passing interface. 
They currently work only on CPU and once the problem is small enough, start conventional multiplication on GPU.


### 05.08.2025:

Fixed constructors when the memory is managed by the mdspan class. (important for usage with the Message passing interface)
Shallow copies now work when the data is managed by the mdspan class 

a sharedptr dummy reference counter was introduced that calls a custom deleter which clears the array and memory mapped files, gpu data if necessary.

(note that in order to achive speed, the element access is always done with raw pointers, the shared ptr is used only in the constructors when the memory is handled by the class).

### 28.07.2025:

Support was added for tensors whose data lies entirely on device.
Fixes for the functions recieving and sending tensors witht he message passing interface was added. (still entirely untested)
Support was added for the message passing interface to send tensors purely to and from device (still entirely untested)


### 01.07.2025:

The library now compiles with optimizations  on gcc 15.1.
With gcc 15.1, the functions of the library can work on the GPU.

It also compiles on clang again, after I removed incorrect code in a function which was not even called by the test application.

Unfortunately, in contrast to gcc 15.1 where it can execute on GPU, clang does not seem to run most functions on the GPU device, even if requested. 
I currently do not know why that is so. The code produces no warnings if compiled with gcc.

In the Cmakelists.txt, -fno-math-errno -fno-trapping-math were added, which speeds the computations up a bit, even if we can not use -O3 currently...


### 30.06.2025:

The library works on gcc 15.1 if no optimizations are switched on.

The Cholesky, LU and QR decompositions now use all available three parallelization levels of the GPU, if the compiler supports them and accepts simd as parallelization level (currently only for gcc).
Also, an initial support for offloading into multiple GPU devices has been added.

With optimizations -O1 of GCC switched on, the compile will currently trigger the following internal compiler error https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120865#add_comment 



The Strassen algorithm only performs the last multiplication on gpu, and if executed on GPU, the Cholesky, LU and QR decomposition do not use the fastest algorithm that is theoretically available (which would use the Strassen algorithm).
One reason reason is that because of this compiler issue https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120753 it is currently difficult to have classes and structs which use device pointers and then run a loop over them. Also, memory copy to the gpu is inherently slow on gpu devices if they are installed in a PCI port. Therefore, the Strassen algorithm is only used for the decompositions if one works on the host, where it should also be possible to use the message passing interface over several nodes (but that is still untested). 

By now, the library also has some support for unified_shared_memory, which, however, is only fast in few (and expensive) nvidia and amd devices https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120679:
More linear algebra routines were added.


### 17.02.2025:

Compilation with Gcc currently produces an internal compiler error due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118590 , https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118738 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118794 . For Windows support, one would have to add Windows specific support for memory mapped files. 


### 13.02.2025:

I fixed a data race which led in some cases to a wrong calculation of the offloaded datalength. now the openmp code yields the same results as the old openacc code.
After running them a few hundreds of times and with different matrices, I can by 13.02.2025, asses that they appear to work correctly if the gpu driver and the cuda version are the most recent.

### 17.02.2025:

The algorithms for the gpu where rewritten such that they now use teams of threads as often as possible. 

Also, initial support for shared memory was added, but I was unable to test it, since my gpu has shared memory but is too old that clang would be able to use it. Due to Openmp's restrictions on the teams distribute pragma, the use of teams of threads is in some cases only possible with shared memory. On some cases with reductions, one unfortunately still has to use threads with parallel for. 
