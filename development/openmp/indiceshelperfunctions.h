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
inline size_t compute_offset(const size_t *  indices,
                             const size_t*  strides,
                             const size_t rank)
{
    size_t offset = 0;
    if constexpr (variant == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd reduction(+ : offset)
        for (size_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }
    else if constexpr (variant == OpenMPVariant::Simd)
    {
        #pragma omp simd reduction(+ : offset)
        for (size_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * strides[i];
        }
    }

    return offset;
}
#pragma omp end declare target



#pragma omp begin declare target
template <OpenMPVariant variant = OpenMPVariant::Sequential>
inline size_t compute_offset(const size_t*  indices,
                               const size_t*  strides_buffer,
                               const size_t rank,
                               const size_t blocknumber)
{

    const size_t* block_strides = strides_buffer + (blocknumber * rank);

    size_t offset = 0;

    if constexpr (variant == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd reduction(+ : offset)
        for (size_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * block_strides[i];
        }
    }
    else if constexpr (variant == OpenMPVariant::Simd)
    {
        #pragma omp  simd reduction(+ : offset)
        for (size_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * block_strides[i];
        }
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < rank; ++i)
        {
            offset += indices[i] * block_strides[i];
        }
    }
    return offset;
}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant variant = OpenMPVariant::Sequential>
inline size_t compute_data_length(const size_t*  extents, const size_t*  strides,const size_t rank)
{
    size_t offset=0;
    if constexpr (variant == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd reduction(+:offset)
        for (size_t i = 0; i < rank; ++i)
        {
            offset += (extents[i]-1) * strides[i];
        }
    }
    else if constexpr (variant == OpenMPVariant::Simd)
    {
        #pragma omp simd reduction(+:offset)
        for (size_t i = 0; i < rank; ++i)
        {
            offset += (extents[i]-1) * strides[i];
        }
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < rank; ++i)
        {
            offset += (extents[i]-1) * strides[i];
        }
    }
    return offset+1;
}
#pragma omp end declare target



#pragma omp begin declare target
inline bool is_row_major(const size_t*extents, const size_t* strides, const size_t rank)
{
    size_t expected = 1;
    for (size_t i = 0; i < rank; ++i)
    {
        if (extents[i] == 1)
            continue;
        if (strides[i] != expected)
            return false;
        expected *= extents[i];
    }
    return true;
}
#pragma omp end declare target


#endif
