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

2) with -O3, -O2 the qr decomposition fails on gpu with nvc++. On these optimization levels, nvc++ appears to try to parallelize loops which are marked as sequential.

3) the qr decomposition is designated as a worker function on device. it this should be able to call vector functions when outside of a parallelized loop. however, when replacing   
T norm = sqrt(gpu_dot_product_s(v,v));
on line 2083 with the vector function
T norm = sqrt(gpu_dot_product_v(v,v));
one gets a sigsev even without -O2 or -O3.

4) On clang, the functions that offload to gpu fail to get called there and can not access the uploaded variables.

