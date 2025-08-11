# AcceleratedLinearAlgebra
A library with some linear algebra functions that works with OpenMP and Open-MPI on the host and on gpu accelerators.

This is a first submission for a simple linear algebra library. It is somewhat an extension to the mdspan class of c++.
While mdspan works with compile time set extents, this library uses c++ concepts, so that stl vectors on the heap can be used for extents.
It has support for rowmajor and column mayor data (but that is not suffciently tested) and rank sizes larger than 2 (but that was not tested very much).

Currently, the library uses open-mp and the message passing interface.

An older version that contains openmp code for the host and open-acc code for the gpu offload is in the archive folder.

The library contains functions for matrix multiplication on accelerators, as well as advanced and fast algorithms from https://arxiv.org/abs/1812.02056 for Cholesky, LU and QR decomposition (besides the usual vector and matrix calculations). The library also has Strassen's algorithm for matrix multiplication implemented, as well as its Winograd Variant from https://arxiv.org/abs/1410.1599 . The algorithms can be set up such that they revert to naive multiplication on host or on gpu when the matrix size is small enough. And the library can work with data from any object with a pointer, including memory mapped files. 

By default, Strassen's algorithm as well as the Winograd variant use memmapped files for temporary data.


The Cholesky, LU and QR decomposition can be set such that they work with multiple cores on CPU and use the gpu only for Matrix multiplication, or they can use Strassen's or Winograds's algorithm for the multiplications. However, the algorithms for Cholesky, LU and QR decomposition, can also work entirely on GPU, using all three parallelization levels that are usually available in these devices (if they are supported by the compiler).

Initial support for the message passing interface was added. The Strassen Algorithm and its Winograd variant can use the Message Passing interface. 

If set up such that one node has a processor and gpu, the algorithms can distribute the multiplication into these submatrices which resist on these nodes. Then, the problem can be offloaded to the gpu. 

This approach may be useful for problems that are too large for a single gpu. A test application may be run with mpirun -np 12 ./arraytest_mpi Be sure to use more or equal nodes than are needed by the recursion. Otherwise cuda will complain maybe because it can not easily start several virtual machines in one process.



A cmakelists.txt file is supplied. 

Version History:

Todo:
1) Let the Strassen and Winograd algorithms work with device pointers for data which is purely located on gpu and then use this in the Cholesky/LU/QR transform
2) Expand the use of the Message Passing Interface to other algorithms.
3) Use this gpu Strassen algorithm and modify the LU, Cholesky, QR decomposition which already work on gpu to use this form of matrix Multiplication on the accellerator instead of the naive version...

Once this is finished:

4) Refractoring: 
Let the mdspan class just have constructors and data management functions, while the datastruct struct has free functions for data management. Put the blas functions as static functions into a friend class of mdspan, so that they can access internal data of mdspan if necessary

5) Then add functions for statistics, function minimization, auto differentiation, optimization, differential equations


By 11.08.2025:
A severe bug was discovered in the Strassen and Winograd algorithms. In order to improve optimization I had added the strides to the () operators of the tensors. This caused difficulties with computations over matrices. I accidentially used two indices, instead of four in a computation of the aforementioned algorithms. This caused wrong results. I now changed this such that the () operators do not need strides. The algorithms now work correctly. I also tested them, in addition to OpenMP, with the Message Passing Interface.

The Strassen and Winograd algorithms now work correctly with the Message Passing interface. 

They can distribute the problem on many nodes, and then, if it is small enough, upload on gpu.
So ideally, one sets up one node per unit consisting of a processor with a gpu. More algorithms for the message passing interface may be added in the future.


By 07.08.2025, 

Some OpenMP shared clauses were fixed,
MPI recieve was put as a constructor into the mdspan class,
MPI send was put in as a method, for the entire class with span fields, and for just the data.
Some Message Passing Interface functions (MPI Send, MPI recieve, MPI Bcast were tested. The test application was updated and now compiles a second application with the OpenMPI replacement compiler.

It can be run with  mpirun -np 12 ./arraytest_mpi 

Unfortunately, the Strassen Algorithm and its Winograd version still have problems and crash when using the Message Passing interface. 
They currently work only on CPU and once the problem is small enough, start conventional multiplication on GPU.


05.08.2025:

Fixed constructors when the memory is managed by the mdspan class. (important for usage with the Message passing interface)
Shallow copies now work when the data is managed by the mdspan class 

a sharedptr dummy reference counter was introduced that calls a custom deleter which clears the array and memory mapped files, gpu data if necessary.

(note that in order to achive speed, the element access is always done with raw pointers, the shared ptr is used only in the constructors when the memory is handled by the class).

By 28.07.2025
Support was added for tensors whose data lies entirely on device.
Fixes for the functions recieving and sending tensors witht he message passing interface was added. (still entirely untested)
Support was added for the message passing interface to send tensors purely to and from device (still entirely untested)




By 01.07.2025 the library now compiles with optimizations  on gcc 15.1.
With gcc 15.1, the functions of the library can work on the GPU.

It also compiles on clang again, after I removed incorrect code in a function which was not even called by the test application.

Unfortunately, in contrast to gcc 15.1 where it can execute on GPU, clang does not seem to run most functions on the GPU device, even if requested. 
I currently do not know why that is so. The code produces no warnings if compiled with gcc.

In the Cmakelists.txt, -fno-math-errno -fno-trapping-math were added, which speeds the computations up a bit, even if we can not use -O3 currently...


By 30.06, the library works on gcc 15.1 if no optimizations are switched on.

The Cholesky, LU and QR decompositions now use all available three parallelization levels of the GPU, if the compiler supports them and accepts simd as parallelization level (currently only for gcc).
Also, an initial support for offloading into multiple GPU devices has been added.

With optimizations -O1 of GCC switched on, the compile will currently trigger the following internal compiler error https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120865#add_comment 



The Strassen algorithm only performs the last multiplication on gpu, and if executed on GPU, the Cholesky, LU and QR decomposition do not use the fastest algorithm that is theoretically available (which would use the Strassen algorithm).
One reason reason is that because of this compiler issue https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120753 it is currently difficult to have classes and structs which use device pointers and then run a loop over them. Also, memory copy to the gpu is inherently slow on gpu devices if they are installed in a PCI port. Therefore, the Strassen algorithm is only used for the decompositions if one works on the host, where it should also be possible to use the message passing interface over several nodes (but that is still untested). 

By now, the library also has some support for unified_shared_memory, which, however, is only fast in few (and expensive) nvidia and amd devices https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120679:

More linear algebra routines were added.






Since 17.02.2025, it runs and compiles with -O3 optimizations switched on.

Compilation with Gcc currently produces an internal compiler error due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118590 , https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118738 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118794 . For Windows support, one would have to add Windows specific support for memory mapped files. 




On 13.02.2025, I fixed a data race which led in some cases to a wrong calculation of the offloaded datalength. now the openmp code yields the same results as the old openacc code.
After running them a few hundreds of times and with different matrices, I can by 13.02.2025, asses that they appear to work correctly if the gpu driver and the cuda version are the most recent.

On 17.02.2025, The algorithms for the gpu where rewritten such that they now use teams of threads as often as possible. 

Also, initial support for shared memory was added, but I was unable to test it, since my gpu has shared memory but is too old that clang would be able to use it. Due to Openmp's restrictions on the teams distribute pragma, the use of teams of threads is in some cases only possible with shared memory. On some cases with reductions, one unfortunately still has to use threads with parallel for. 
