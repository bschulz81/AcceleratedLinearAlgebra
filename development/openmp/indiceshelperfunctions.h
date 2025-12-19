#ifndef HELPERFUNCTIONS
#define HELPERFUNCTIONS

#include "omp.h"
#include <cstddef>



#pragma omp begin declare target
inline size_t compute_offset(const size_t row, const size_t col, const size_t* __restrict  strides)
{
    return row * strides[0]+col*strides[1];
}
#pragma omp end declare target

#pragma omp begin declare target
inline size_t compute_offset(const size_t row, const size_t col, const size_t matrixstride0, const size_t matrixstride1)
{
    return row * matrixstride0+col*matrixstride1;
}
#pragma omp end declare target

#pragma omp begin declare target
inline size_t compute_offset_w(const size_t * __restrict indices, const size_t* __restrict strides,const size_t r)
{
    size_t offset = 0;
    // Row-major layout: iterate outermost to innermost
    #pragma omp parallel for simd reduction(+ : offset)
    for (size_t i = 0; i < r; ++i)
    {
        offset += indices[i] * strides[i];
    }

    return offset;
}
#pragma omp end declare target

#pragma omp begin declare target
inline size_t compute_offset_s(const size_t * __restrict indices, const size_t* __restrict strides,const size_t r)
{
    size_t offset = 0;
    // Row-major layout: iterate outermost to innermost
    #pragma omp unroll partial
    for (size_t i = 0; i < r; ++i)
    {
        offset += indices[i] * strides[i];
    }
    return offset;
}
#pragma omp end declare target




#pragma omp begin declare target
inline size_t compute_offset_v(const size_t * __restrict indices, const size_t* __restrict strides,const size_t r)
{
    size_t offset = 0;

    // Row-major layout: iterate outermost to innermost
    #pragma omp simd reduction(+ : offset)
    for (size_t i = 0; i < r; ++i)
    {
        offset += indices[i] * strides[i];
    }
    return offset;
}
#pragma omp end declare target



#pragma omp begin declare target
inline size_t compute_data_length_w(const size_t*__restrict  extents, const size_t*__restrict  strides,const size_t rank)
{

    size_t offset=0;
    #pragma omp parallel for simd reduction(+:offset)
    for (size_t i = 0; i < rank; ++i)
    {
        offset += (extents[i]-1) * strides[i];
    }
    return offset+1;
}
#pragma omp end declare target

#pragma omp begin declare target
inline size_t compute_data_length_v(const size_t*__restrict  extents, const size_t*__restrict  strides,const size_t rank)
{
    size_t offset=0;
    #pragma omp simd reduction(+:offset)
    for (size_t i = 0; i < rank; ++i)
    {
        offset += (extents[i]-1) * strides[i];
    }
    return offset+1;
}
#pragma omp end declare target



#pragma omp begin declare target
inline size_t compute_data_length_s(const size_t*__restrict  extents, const size_t*__restrict  strides,const size_t rank)
{
    size_t offset=0;
    for (size_t i = 0; i < rank; ++i)
    {
        offset += (extents[i]-1) * strides[i];
    }
    return offset+1;
}
#pragma omp end declare target



#endif
