The files here are an initial openmp port of the openacc code.

Unfortunately, by now this is in the development folder for the following reasons:


Unfortunately, this code fails on clang at runtime with optimizations set on, see the compiler bug which I filed.
https://github.com/llvm/llvm-project/issues/126342  It seems to have something to do with openmp optimizations:

Also, In order to make the qr-decomposition work on gpu, i even had to remove some parallel for statements, and remove simd loops.


Compilation with nvc++ seems to fail because nvc++ does not recognize #pragma omp begin declare target commands, even though this is openmp standard...
nvc++ claims: C++-S-0155-No matching #pragma omp declare target found for current end declare target directive  (/home/benni/projects/arraylibrary/openmp/mdspan_omp.h)
(indeed, that is because one has to write #pragma omp begin declare target and i am not making my code non standard compilant.. omp declare target is nore for single variables not regions)
 
gcc compilation fails due to the following compiler bugs  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118794



