Todo:
Version History:

Todo:

1) Let the Strassen and Winograd algorithms work with device pointers for data which is purely located on gpu and then use this in the Cholesky/LU/QR transform
2) Expand the use of the Message Passing Interface to other algorithms.
3) Use this gpu Strassen algorithm and modify the LU, Cholesky, QR decomposition which already work on gpu to use this form of matrix Multiplication on the accellerator instead of the naive version...
Once this is finished:

4) Refractoring: Let the mdspan class just have constructors and data management functions, while the datastruct struct has free functions for data management. Put the blas functions as static
functions into a friend class of mdspan, so that they can access internal data of mdspan if necessary

5) Then add functions for statistics, function minimization, auto differentiation, optimization, differential equations

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



