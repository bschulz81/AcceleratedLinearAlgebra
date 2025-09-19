Todo:
Version History:

Todo:
1) Test the message passing interface support
2) Add options for the linear algebra functions such that most of them can use the message passing interface as well as the gpu then for local work.
3) add functions for statistics, function minimization, auto differentiation, optimization, differential equations



By 19.09.2025
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



By 17.09.25;
Updated the main_mpi.cpp file to use the new printtensor() function instead of the removed printmatrix function, 
The printtensor function prints tensors residing on host as well as on device and can work with tensors of all ranks.
By 16.09.25
Speed improvements in the datastruct class for the supspanmatrix, subspan, row and column extraction methods, and the matrix multiplication.
The column and row methods have now rank reducing and rank preserving forms
The printtensor method now works with device data too
The mdspan class can now work with device pointers.
The mdspan class has a mapping manager that is shared among instances and provides book keeping to prevent overlapping mappings, which are forbidden in the openmp standard.
The mdspan data class can now create data purely on device. 
The code of the mdspan and mdspan_data and datastruct classes was significantly polished, improved and tested for various parameters and circumstances.
The test application demonstrates more basic matrix and tensor access on device and on host.



By 09.09.25
Fixed the class in datastruct.h to accomodate for column major matrices in addition to the rowmajor case which was used in the algorithms earlier. I added test cases for column major data

By 06.09.25
Fixed a bug in the GPU version of the advanced algorithm for Cholesky decomposition.
All the decomposition algorithms from  https://arxiv.org/pdf/1812.02056 as well as the Strassen algorithm and its Winograd Variant now work on device https://arxiv.org/abs/1410.1599



By 06.09.25,
1) the advanced algorithms for LU/and QR decomposition as well as the Strassen and Winograd algorithms can now work purely on gpu with gpu data pointers.

2) the advanced algorithm for the QR decomposition  from https://arxiv.org/pdf/1812.02056 showed severe numerical stability errors. But these are inherent in the algorithms
from that paper. I have included some measures to increase stability. The instability arises because the advanced algorithms use the Strassen algorithm twice for one
matrix multiplication after another and then a Grahm Schmidt orthonormalization procedure.
The Strassen algorithm replaces multiplications by faster additions, which are, however, numerically unstable.
The Grahm Schmidt, and any other orthonormalization procedure uses dot products that involve large sums over columns of matrices. These are also numerically unstable.

So the algorithm employs three numerically unstable methods in a chain.

For my test data, I found that it could be stabilized a bit by replacing one Strassen multiplication
by an ordinary one. However, given that the error becomes larger with larger sums, i.e. larger matrices, I need to test stability with a larger matrix. 
Of course the library is also able to use the simple algorithms on gpu, which are not affected by stability problems from Strassen multiplication, but any QR decomposition
needs dot products of vectors and is affected by numerical instability of large sums. In order to increase precision, I have began to add methods for Kahan sums for products.


By 31.08, the changes were as follows:
fixed a bug in the matrix*vector multiply function. 
broke the library down into several classes in different files, which are easily testable. 
one basic datastruct class, on which mathematical function can operate and which can be offloaded to gpu, one mdspan child class which can host strides and extents, 
another childclass clalled mdspan_data, which hosts the data as well. Additionally a policy model for the mathematical algorithms. This policy model is able to autoconfigure options and 
can decide automatically, whether the function should offload to gpu, or whether the message passing interface is used in the Strassen algorithm

separated the mathematical funcitons such that they could be easily replaced by fortran functions for improved speed.
The strassen algorithm, as well as the advanced algorithms for cholesky, lu and qr decomposition now work on GPU, unfortunately only in unified shared memory mode, due to the recursive nature.
added more message passing interface support. But most of that is untested until now.


By 11.08.2025: A severe bug was discovered in the Strassen and Winograd algorithms. In order to improve optimization I had added the strides to the () operators of the tensors. 
This caused difficulties with computations over matrices. I accidentially used two indices, instead of four in a computation of the aforementioned algorithms.
This caused wrong results. I now changed this such that the () operators do not need strides. The algorithms now work correctly. 
I also tested them, in addition to OpenMP, with the Message Passing Interface.

The Strassen and Winograd algorithms now work correctly with the Message Passing interface.

They can distribute the problem on many nodes, and then, if it is small enough, upload on gpu. So ideally, one sets up one node per unit consisting of a processor with a gpu.
More algorithms for the message passing interface may be added in the future.

By 07.08.2025,

Some OpenMP shared clauses were fixed, MPI recieve was put as a constructor into the mdspan class, MPI send was put in as a method, for the entire class with span 
fields, and for just the data. Some Message Passing Interface functions (MPI Send, MPI recieve, MPI Bcast were tested. The test application was updated and now
compiles a second application with the OpenMPI replacement compiler.

It can be run with mpirun -np 12 ./arraytest_mpi

Unfortunately, the Strassen Algorithm and its Winograd version still have problems and crash when using the Message Passing interface. 
They currently work only on CPU and once the problem is small enough, start conventional multiplication on GPU.



05.08.2025:
Fixed constructors when the memory is managed by the mdspan class. (important for usage with the Message passing interface)
Shallow copies now work when the data is managed by the mdspan class 
a sharedptr dummy reference counter was introduced that calls a custom deleter which clears the array and memory mapped files, gpu data if necessary.
(note that in order to achive speed, the element access is always done with raw pointers, the shared ptr is used only in the constructors when the memory is handled by the class).


28.07.2025:
Support was added for tensors whose data lies entirely on device.

Fixes for the functions recieving and sending tensors witht he message passing interface was added. (still entirely untested)

Support was added for the message passing interface to send tensors purely to and from device (still entirely untested)



01.07.2025:
I fixed several functions which were not called by the small test application. Now the code compiles with gcc 15.1 even if optimizations are switched on.
It also compiles with clang again. Unfortunately, clang does not seem to execute many of the loops on the gpu device, even if requested.
In contrast, the code produced by gcc starts cuda kernels and runs on my GPU device

30.06.2025:

More linear algebra routines were added
Initial support for unified shared memory was added

The Cholesky, LU, LU factorizations now use all 3 parallelization levels that gpu devices usually have

Initial support for offloading to several gpu devices added
Compiles with gcc 15.1 if no optimizations are enabled.


Compilation with clang will trigger an internal compiler error with the QR factorization https://github.com/llvm/llvm-project/issues/146262
Compilation with -O1 with gcc will trigger an interla compiler error https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120865

No support for Strassen's algorithm on the gpu, as it needs many temporary files. These data would be organized in structs,
but using structs and classes together with data that rests solely on gpu is prevented by the following compiler problem https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120753



The files here are an initial openmp port of the openacc code.

On 13.02, I found a data race in compute_datalength. 
After fixing this, the code produces correct results everytime. 

On 17.02.2025, The algorithms for the gpu where rewritten such that they now use teams of threads as often as possible. 
The rewrite took Openmp's restrictions for the teams distribute pragma into account.

Also, initial support for shared memory was added, but I was unable to test it, since my gpu has shared memory but is too old that clang would be able to use it. 
Due to Openmp's restrictions on the teams distribute pragma, the use of teams of threads is in some cases only possible with shared memory. 

On some cases with reductions, one unfortunately still has to use threads with parallel for. 

The library now compiles with -O3 with clang.



Compilation with nvc++ seems to fail because nvc++ does not recognize #pragma omp begin declare target commands, even though this is openmp standard...

gcc compilation fails due to the following compiler bugs  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118794



