
30.06:

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



