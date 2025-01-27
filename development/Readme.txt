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

2) Unfortunately, if compiled, it appears that the cholesky decomposition fails if it should work entirely on gpu and one compiles with -O2 or -O3 and nvc++
  
3) the new/delete[] calls in gpu_lu_decomposition, gpu_cholesky_decomposition, gpu_qr_decomposition should be replaced by acc_alloc/acc_free. Unfortunately, 
  on my system, nvc++ does not link to the library where these files are, despite I am able to use other acc functions.
