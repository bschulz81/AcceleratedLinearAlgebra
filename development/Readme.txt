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

The open_acc standard says that functions designated as workers can call functions which have a worker loop.

However, nvc++ apparently had problems to call these functions from a sequential loop. So they were inserted in plain text. Now these loops are parallelized. However, nvc++ often sees data dependencies, where there clearly are none, and refuses to vectorize. 

This is a problem that may or may not be solved by suitable code refractorings...


4) On clang, the functions that offload to gpu fail, unfortunately.


