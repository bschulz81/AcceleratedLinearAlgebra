The files in this directory are development code.
They are not really tested and do not necessarily work correctly. Use at your own risk.

An initial support for the message passing interface was added in the Strassen and Winograd Matrix multiplication algorithms. But this was not  tested yet in any way.

Updates:
The structs for the matrix multiplication parameters were slightly changed, in order to account for the message passing interface and more flexibility with open-mp.
In order to prevent openmp to possibly overfill the gpu memory by uploading large matrices in parallel,
the user can now change whether openmp should be used in the steps where the winograd or strassen matrix multiplications start sub-processes for the multiplication. That way,

One may, for example, configure the message passing interface such that there is exactly one node per host with a gpu. then if openmp, and gpu offload is used in strassen's algorithm, 
(and if the untested mpi support works as designed), Strassen's algorithm will, work parallel with several cpu's for the additions.
It  will also do the remaining multiplications in parallel if the Open_MPI computers have all been send a sub matrix, or if a certain size has been reached where openmp should be used. 

If the matrices are small enough, they can also be uploaded to the gpu for naive computation, and when that is finished, they are send back from the gpu to the smaller ranks to be combined.

Some identified Issues:

1) The initial support for the message passing interface was not tested yet in any way.

2) The Cholesky, LU, and QR decompositions have the matrix multiplications and dot products that had to be inlined in plain.  The open_acc standard says that functions designated as workers can call procedures which have a worker loop or contain vector loops. However, nvc++ apparently has problems to call these functions (even vector functions) from a sequential loop of a worker function. After I inserted these internal loops in plain text, they are now parallelized.

3) Additionally, nvc++ often sees data dependencies in simple loops, which do not seem to be there and refuses to vectorize or to parallelize. 
On 03.02.2025, the openacc loops were set with the "independent" clause such that nvc++ can vectorize more openacc loops. Unfortunately, openmp does not have such a clause.  As of 04.02.2025, I have used the __restrict and const keyword where ever useful. I also added the option  -Msafeptr=all to the Cmakelists.txt for nvc++. This seems to remove some of the pointer overlapping assumptions and now some optimizations can finally take place but there are still issues. sometimes nvc++ refuses to vectorize openmp code for reasons it calls "unknown" or for "data dependencies that are not relevant to the computation, or are not there. Some loops are not parallelized with reasons given as "not vectorized because unknown" from nvc++. On some loops nvc++ makes parallelization attempts even if no acc clause was added.  

4) I have yet to figure out how to turn the LU/Cholesky/QR decompositions into gang routines. Unfortunately, during attempts to make the matrix multiplications gang loops in a gang routine, they turned out to yield wrong and different numbers at each run. 

By now (on 12.02.2025), after running a few hundred tests with several matrices, i think I can now asses that the worker loop versions of the QR/Cholesky/LU decomposition work correctly.

5) Unfortunately, gcc and clang have difficulties with the open-acc and open-mp offload compilers, which are still in development. Gcc compilation currently fails because of https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118738 , https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118590 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518 .

On clang, the functions that offload to gpu fail, unfortunately, due to openacc being in deelopment.

6) I have added an openmp branch. This is for development purposes only. 

On 13.02.2025, I fixed a bug in the openmp branch where a data race often caused wrong results for the length of the data to be uploaded. After this is fixed, the openmp version now works and compiles with clang. Due to a clang bug in the optimizer (which seems to be  a problem with a parallelizable loop within a non-parallelizable for loop), one can not compile the code with optimizations turned on. Without optimizations, the code would run fine and yield correct results. With optimizations turned on, clang's code would cause a memory segfault at runtime.


