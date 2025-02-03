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

However, nvc++ apparently had problems to call these functions from a sequential loop. It does clearly not correspond to the openacc standard that a function designated as worker, which contains a worker loop can not call a vector loop. After I inserted these internal loops in plain text, they are parallelized.  

Additionally, nvc++ often saw data dependencies in simple loops, where there are none, and refused to vectorize. More precisely, nvc++ is often confused by the matrix strides if no independent clause is added to the loops. 
By now, (03.02.2025), the openacc pragmas were set such that nvc++ can vectorize more openacc loops.

Unfortunately, in contrast to openacc, openmp has no "independent" clause for loops. As a result, some openmp loops for the host are not vectorized with nvc++, sometimes this comes with the comment "not vectorized because unknown" from nvc++. 

This seems to be different from other compilers, like gcc or clang, which can vectorize such code. Unfortunately, gcc and clang have difficulties with the open-acc and openmp offload by now, which is still in development for these open-source compilers. 

4) On clang, the functions that offload to gpu fail, unfortunately.


