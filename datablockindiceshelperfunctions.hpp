#ifndef HELPERFUNCTIONS
#define HELPERFUNCTIONS

#include "omp.h"
#include <limits.h>
#include <complex>
#include <print>
#include<iostream>

#pragma omp begin declare target
inline void fill_strides(const ptrdiff_t*    extents,ptrdiff_t*    strides, const ptrdiff_t rank, const bool rowmajor)
{
    if (rank==0)
        return;

    if (rowmajor)
    {

        strides[rank - 1] = 1;
        #pragma omp unroll partial
        for (int i = rank - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
    else
    {
        strides[0] = 1;
        #pragma omp unroll partial
        for (ptrdiff_t i = 1; i < rank; ++i)
        {
            strides[i] = strides[i - 1] * extents[i - 1];
        }
    }
}

#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
struct is_complex : std::false_type {};
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
inline constexpr auto cond_conj(const T& val)
{
    if constexpr (is_complex<T>::value)
    {
        return std::conj(val);
    }
    else
    {
        return val;
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
inline constexpr auto returnval(const T& val,bool conj)
{
    if constexpr (is_complex<T>::value)
    {
        return conj? std::conj(val):val;
    }
    else
    {
        return val;
    }
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T, typename = std::void_t<>>
struct has_print : std::false_type {};
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
struct has_print<T, std::void_t<decltype(std::declval<T>().print())>> : std::true_type {};
#pragma omp end declare target

#pragma omp begin declare target
template <typename T, typename = std::void_t<>>
struct has_buffer_print : std::false_type {};
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
struct has_buffer_print<T, std::void_t<
decltype(std::declval<T>().print_to_buffer(std::declval<char*>(), std::declval<ptrdiff_t>())),
decltype(std::declval<T>().required_buffer_size())
>> : std::true_type {};
#pragma omp end declare target





#pragma omp declare reduction(+: std::complex<double>: \
omp_out += omp_in) \
initializer(omp_priv(0.0, 0.0))

#pragma omp declare reduction(+: std::complex<float>: \
omp_out += omp_in) \
initializer(omp_priv(0.0f, 0.0f))

#pragma omp declare reduction(+: std::complex<long double>: \
omp_out += omp_in) \
initializer(omp_priv(0, 0))


#pragma omp begin declare target
template <typename T>
void print_variable(const T& var,bool conjugate)
{

    if constexpr (is_complex<T>::value)
    {
        double real_part = static_cast<double>(var.real());
        double imag_part = static_cast<double>(var.imag());
        if(conjugate)
            printf("(%g, %g)", real_part, -imag_part);
        else
            printf("(%g, %g)", real_part, imag_part);

    }

    else if constexpr (std::is_floating_point_v<T>)
    {
        printf("%g", static_cast<double>(var));
    }

    else if constexpr (std::is_integral_v<T>)
    {
        printf("%lld", static_cast<long long>(var));
    }

    else if constexpr (has_print<T>::value)
    {
        var.print();
    }
    else
    {
        printf("[Unknown Object]");
    }
}
#pragma omp end declare target




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
