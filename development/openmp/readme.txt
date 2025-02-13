The files here are an initial openmp port of the openacc code.

Unfortunately, by now this is in the development folder for the following reasons:


Unfortunately, this code fails on clang at runtime with optimizations set on, see the compiler bug which I filed.
https://github.com/llvm/llvm-project/issues/126342  It seems to have something to do with openmp optimizations:

On 13.02, I found a data race in compute_datalength. After fixing this, the code produces correct results everytime. but the bug in clangs optimizer currently forbids compilation at higher o levels.
However, the teams versions of the loops now work in the openmp implementation, which should yield a speed increase. Unfortunately, clang does not yet support openmp simd loops on the gpu.


Compilation with nvc++ seems to fail because nvc++ does not recognize #pragma omp begin declare target commands, even though this is openmp standard...

gcc compilation fails due to the following compiler bugs  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118518 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=118794



