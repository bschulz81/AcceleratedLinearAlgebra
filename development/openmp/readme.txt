Todo:

Test of the tensor library with the message passing interface.
Implementation of the strassen algorithm with device pointers and cuda aware message passing interface for tensors purely on device.
Then use this strassen algorithm and modify the LU, Cholesky, QR decomposition for the gpu to use this version...

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

No support for Strassen's algorithm on the gpu, as it needs many temporary files. These data would be organized in structs, but using structs and classes together with data that rests solely on gpu is prevented by the following compiler problem https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120753



The files here are an initial openmp port of the openacc code.

On 13.02, I found a data race in compute_datalength. 
After fixing this, the code produces correct results everytime. 

On 17.02.2025, The algorithms for the gpu where rewritten such that they now use teams of threads as often as possible. 
The rewrite took Openmp's restrictions for the teams distribute pragma into account.

Also, initial support for shared memory was added, but I was unable to test it, since my gpu has shared memory but is too old that clang would be able to use it. Due to Openmp's restrictions on the teams distribute pragma, the use of teams of threads is in some cases only possible with shared memory. 

On some cases with reductions, one unfortunately still has to use threads with parallel for. 

The library now compiles with -O3 with clang.



Compilation with nvc++ seems to fail because nvc++ does not recognize #pragma omp begin declare target commands, even though this is openmp standard...

gcc compilation fails due to the following compiler bugs  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118794



