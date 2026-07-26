#ifndef HELPERFUNCTIONS
#define HELPERFUNCTIONS

#include "omp.h"

#pragma omp begin declare target
enum class OpenMPVariant
{
    ParallelSimd,
    Simd,
    Sequential
};
#pragma omp end declare target

#pragma omp begin declare target
template <OpenMPVariant variant = OpenMPVariant::Sequential>
inline ptrdiff_t compute_offset(const ptrdiff_t *  indices,
                             const ptrdiff_t*  strides,
                             const ptrdiff_t rank)
{
    ptrdiff_t offset = 0;
    if constexpr (variant == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd reduction(+ : offset)
        for (ptrdiff_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }
    else if constexpr (variant == OpenMPVariant::Simd)
    {
        #pragma omp simd reduction(+ : offset)
        for (ptrdiff_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }
    else
    {
        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }

    return offset;
}
#pragma omp end declare target



#pragma omp begin declare target
template <OpenMPVariant variant = OpenMPVariant::Sequential>
inline ptrdiff_t compute_offset(const ptrdiff_t*  indices,
                               const ptrdiff_t*  strides_buffer,
                               const ptrdiff_t rank,
                               const ptrdiff_t blocknumber)
{

    const ptrdiff_t* block_strides = strides_buffer + (blocknumber * rank);

    ptrdiff_t offset = 0;

    if constexpr (variant == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd reduction(+ : offset)
        for (ptrdiff_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * block_strides[i];
        }
    }
    else if constexpr (variant == OpenMPVariant::Simd)
    {
        #pragma omp  simd reduction(+ : offset)
        for (ptrdiff_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * block_strides[i];
        }
    }
    else
    {
        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * block_strides[i];
        }
    }
    return offset;
}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant variant = OpenMPVariant::Sequential>
inline ptrdiff_t compute_data_length(const ptrdiff_t*  extents, const ptrdiff_t*  strides,const ptrdiff_t rank)
{
    ptrdiff_t offset=0;
    if constexpr (variant == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd reduction(+:offset)
        for (ptrdiff_t i = 0; i < rank; ++i)
        {
            offset += (extents[i]-1) * strides[i];
        }
    }
    else if constexpr (variant == OpenMPVariant::Simd)
    {
        #pragma omp simd reduction(+:offset)
        for (ptrdiff_t i = 0; i < rank; ++i)
        {
            offset += (extents[i]-1) * strides[i];
        }
    }
    else
    {
        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < rank; ++i)
        {
            offset += (extents[i]-1) * strides[i];
        }
    }
    return offset+1;
}
#pragma omp end declare target



#pragma omp begin declare target
inline bool is_row_major(const ptrdiff_t*extents, const ptrdiff_t* strides, const ptrdiff_t rank)
{
    ptrdiff_t expected = 1;
    for (ptrdiff_t i = 0; i < rank; ++i)
    {
        if (extents[i] <= 1)
            continue;
        if (abs(strides[i]) != expected)
            return false;
        expected *= extents[i];
    }
    return true;
}
#pragma omp end declare target


#endif
