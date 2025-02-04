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
If the matrices are small enogh, they can also be uploaded to the gpu for naive computation, and when that is finished, they are send back from the gpu to the smaller ranks to be combined.

Some identified Issues:

1) The initial support for the message passing interface was not tested yet in any way.

2) The cholesky, lu, and qr decompositions have the matrix multiplications and dot products now inlined in plain. 


The open_acc standard says that functions designated as workers can call procedures which have a worker loop or contain vector loops.

However, nvc++ apparently had problems to call these functions (even vector functions) from a sequential loop of a worker function. It does clearly not correspond to the openacc standard that a function designated as worker, which contains a sequential loop can not call a function with a vector loop. After I inserted these internal loops in plain text, they are now parallelized.  

Additionally, nvc++ often saw data dependencies in simple loops, where there are none, and refused to vectorize. More precisely, nvc++ is often confused by the matrix strides if no independent clause is added to the loops. 
By now, (03.02.2025), the openacc pragmas were set such that nvc++ can vectorize more openacc loops.

As of 04.02, i have used the __restrict and const keyword where ever useful. Apparently, by default nvc++ assumes that the pointers overlap (which i find quite strange, since this implies that the assumption is that a programmer writes a vector loop without knowing when this is possible). 

If given the __restrict keyword, the gpu code is now much much faster and gets out instantly. 

(Most warnings of non-parallelizable code are now issued for functions and loops where no parallelization should happen (e.g. a printmatrix function, which is also strange. Why does the compiler try to parallelize code where no openacc or openmp pragma appeared). But there are still some issues with openmp loops. Openmp has no "independent" clause as it assumes the programmer to know what he does. Sometimes nvc++ refuses to vectorize openmp code for reasons it calls "unknown" or for "data dependencies that clearly are not there. Sometimes this comes with the comment "not vectorized because unknown" from nvc++.  

This seems to be different from other compilers, like gcc or clang, which can vectorize such code. On 04.02.25, I now added the option  -Msafeptr=all to the Cmakelists.txt for nvc++. This seems to remove some of the pointer overlapping assumptions and now some optimizations can finally take place.


Unfortunately, gcc and clang have difficulties with the open-acc and openmp offload by now, which is still in development for these open-source compilers. Gcc compilation currently fails because of https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118738 , https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118590 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518 .

Openacc support is better developed in nvc++. The matrix multiplication can be parallelized with openacc and nvc++ on the device but nvc++ refuses to parallelize the exactly same code with open-mp pragmas for the host.

Also, apparently, from functions denoted as worker, if they have a sequential loop, one can not call any other parallelized functions, including those denoted as vectors in nvc++ currently. Therefore, the matrix multiplications in the Cholesky/LU/QR decompositions within these loops had to be inlined. Also, vector loops in the constructors of the datastruct class were removed in order to prevent crashes. In the current applications, these loops would not even be called, but for tensors, this could lead to a slower creation, since some of the loops in these constructors for tensors with rank>2 could be parallelized in theory.



4) On clang, the functions that offload to gpu fail, unfortunately.


