#ifndef INKERNELMATHFUNCTIONS
#define INKERNELMATHFUNCTIONS

#include "cmath"
#include "datablock.h"
#include "datablockcontainer.h"
#include "datablockutilities.h"
#include "host_memory_functions.h"
using namespace std;



#pragma omp begin declare target

class In_Kernel_Mathfunctions
{
public:
    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void cholesky_decomposition(const DataBlock<T>& A, DataBlock<T>& L,bool initialize_to_zero=true);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void lu_decomposition(const  DataBlock<T>& dA, DataBlock<T>& dL, DataBlock<T>& dU,bool initialize_to_zero=true);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void qr_decomposition( const DataBlock<T>&A, DataBlock<T> Q, DataBlock<T> &R,bool initialize_to_zero=true,bool with_memmaps=false);

    template <typename T>
    inline static void cross_product( const DataBlock<T>& vec1,const   DataBlock<T>& vec2, DataBlock<T>& res);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot( const DataBlock<T>& A,  const DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot_kahan(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot_accumulate( const DataBlock<T>& A,  const DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot_accumulate_kahan(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_add( const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_add_accumulate( DataBlock<T>& A,const DataBlock<T>& B);


    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_subtract(const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_subtract_accumulate( DataBlock<T>& A,const  DataBlock<T>& B);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_vector( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T> &C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_vector_kahan( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_vector( const DataBlock<T>&M,const  T*V, DataBlock<T> & C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_vector_kahan( const DataBlock<T>&M,const  T*V, DataBlock<T> & C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_add(const  DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_add_accumulate(  DataBlock<T>& vec1,const  DataBlock<T>& vec2);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_subtract(const  DataBlock<T>& vec1, const  DataBlock<T>& vec2, DataBlock<T> & res);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_subtract_accumulate(  DataBlock<T>& vec1,const  DataBlock<T>& vec2);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static T dot_product( const DataBlock<T> &vec1,const  DataBlock<T> &vec2);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static T dot_product_kahan(const  DataBlock<T> &vec1, const DataBlock<T> &vec2);



    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_scalar( const  DataBlock<T>& M, const T V, DataBlock<T>& C);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_scalar_accumulate(   DataBlock<T>& M, const T V );

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_multiply_scalar( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void vector_multiply_scalar_accumulate(  DataBlock<T>& vec,const T scalar);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot_sparse(const BlockedDataView<T>& Ablocks, const BlockedDataView<T>& Bblocks, DataBlock<T>& C,bool initialize_output_to_zero=true);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static void matrix_multiply_dot_sparse(const BlockedDataView<T>& Ablocks, const DataBlock<T>& Bblocks, DataBlock<T>& C,bool initialize_output_to_zero=true);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static  void matrix_vector_multiply_sparse(const BlockedDataView<T>& A, const DataBlock<T>& x,  DataBlock<T>& y,bool initialize_output_to_zero=true);

    template <OpenMPVariant Policy=OpenMPVariant::ParallelSimd, typename T>
    inline static  void matrix_vector_multiply_sparse(const BlockedDataView<T>& A, const BlockedDataView<T>& x, DataBlock<T>& y,bool initialize_output_to_zero=true);

    template <typename T>
    inline static T kahan_sum(const T *arr,size_t n);

    template <typename T>
    inline static T neumaier_sum(const T*arr,size_t n);
};
#pragma omp end declare target

#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_vector_multiply_sparse(  const BlockedDataView<T>& A, const BlockedDataView<T>& x,  DataBlock<T>& y,bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;
    const size_t nblocks = x.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];
    const size_t Xblock_size = x.block_shape[0];


    const size_t aext0 = A.dpextents[0];
    const size_t aext1 = A.dpextents[1];
    const size_t xext  = x.dpextents[0];

    const size_t ystr0 = y.dpstrides[0];

    if(initialize_to_zero)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp parallel for simd
            for(size_t i=0; i<y.dpextents[0]; i++)
                y.dpdata[i*ystr0]=T(0);
        }
        else  if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp  simd
            for(size_t i=0; i<y.dpextents[0]; i++)
                y.dpdata[i*ystr0]=T(0);
        }
        else
        {
            #pragma omp  unroll partial
            for(size_t i=0; i<y.dpextents[0]; i++)
                y.dpdata[i*ystr0]=T(0);
        }
    }

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for collapse(2)
        for (size_t ia = 0; ia < mblocks; ++ia)
        {
            for (size_t jb = 0; jb < nblocks; ++jb)
            {
                const size_t a_start = A.pooled_offsets_starts[ia];
                const size_t* a_off  = A.pooled_offsets_flat + a_start;

                const size_t a_row_off = a_off[0];
                const size_t a_col_off = a_off[1];

                const size_t a_rem_rows = aext0 - a_row_off;
                const size_t a_rem_cols = aext1 - a_col_off;

                const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
                const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

                const size_t x_start = x.pooled_offsets_starts[jb];
                const size_t* x_off  = x.pooled_offsets_flat + x_start;

                const size_t x_off0 = x_off[0];
                const size_t x_rem  = xext - x_off0;
                const size_t x_tile = (Xblock_size < x_rem) ? Xblock_size : x_rem;

                const size_t k_start = (a_col_off> x_off0) ? a_col_off:x_off0;
                const size_t a= a_col_off + a_tile_cols;
                const size_t b=x_off0 + x_tile;
                const size_t k_end   =(a<b)?a:b;

                if (k_start >= k_end) continue;

                for (size_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const size_t global_i = a_row_off + ii;
                    T sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t kk = k_start; kk < k_end; ++kk)
                    {

                        sum += A(global_i,kk)* x(kk);
                    }
                    #pragma omp atomic update
                    y(global_i) += sum;
                }
            }
        }
    }

    if constexpr (Policy == OpenMPVariant::Simd)
    {


        for (size_t ia = 0; ia < mblocks; ++ia)
        {
            const size_t a_start = A.pooled_offsets_starts[ia];
            const size_t* a_off  = A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];

            const size_t a_rem_rows = aext0 - a_row_off;
            const size_t a_rem_cols = aext1 - a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            const size_t a= a_col_off + a_tile_cols;
            for (size_t jb = 0; jb < nblocks; ++jb)
            {


                const size_t x_start = x.pooled_offsets_starts[jb];
                const size_t* x_off  = x.pooled_offsets_flat + x_start;

                const size_t x_off0 = x_off[0];
                const size_t x_rem  = xext - x_off0;
                const size_t x_tile = (Xblock_size < x_rem) ? Xblock_size : x_rem;

                const size_t k_start = (a_col_off> x_off0) ? a_col_off:x_off0;

                const size_t b=x_off0 + x_tile;
                const size_t k_end   =(a<b)?a:b;

                if (k_start >= k_end) continue;
                for (size_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const size_t global_i = a_row_off + ii;
                    T sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t kk = k_start; kk < k_end; ++kk)
                    {

                        sum += A(global_i,kk)* x(kk);
                    }
                    y(global_i) += sum;
                }
            }
        }
    }
    else
    {

        for (size_t ia = 0; ia < mblocks; ++ia)
        {
            const size_t a_start = A.pooled_offsets_starts[ia];
            const size_t* a_off  = A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];

            const size_t a_rem_rows = aext0 - a_row_off;
            const size_t a_rem_cols = aext1 - a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            const size_t a= a_col_off + a_tile_cols;

            for (size_t jb = 0; jb < nblocks; ++jb)
            {


                const size_t x_start = x.pooled_offsets_starts[jb];
                const size_t* x_off  = x.pooled_offsets_flat + x_start;

                const size_t x_off0 = x_off[0];
                const size_t x_rem  = xext - x_off0;
                const size_t x_tile = (Xblock_size < x_rem) ? Xblock_size : x_rem;

                const size_t k_start = (a_col_off> x_off0) ? a_col_off:x_off0;

                const size_t b=x_off0 + x_tile;
                const size_t k_end   =(a<b)?a:b;

                if (k_start >= k_end) continue;
                for (size_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const size_t global_i = a_row_off + ii;
                    T sum = 0;
                    #pragma omp unroll partial
                    for (size_t kk = k_start; kk < k_end; ++kk)
                    {
                        sum += A(global_i,kk)* x(kk);
                    }
                    y(global_i) += sum;
                }
            }
        }
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_vector_multiply_sparse( const BlockedDataView<T>& A, const DataBlock<T>& x, DataBlock<T>& y,bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];



    const size_t aext0 = A.dpextents[0];
    const size_t aext1 = A.dpextents[1];

    const size_t ystr0 = y.dpstrides[0];

    if(initialize_to_zero)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp parallel for simd
            for(size_t i=0; i<y.dpextents[0]; i++)
                y.dpdata[i*ystr0]=T(0);
        }
        else  if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp simd
            for(size_t i=0; i<y.dpextents[0]; i++)
                y.dpdata[i*ystr0]=T(0);
        }
        else
        {
            #pragma omp  unroll partial
            for(size_t i=0; i<y.dpextents[0]; i++)
                y.dpdata[i*ystr0]=T(0);
        }
    }


    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for
        for (size_t ia = 0; ia < mblocks; ++ia)
        {

            const size_t a_start = A.pooled_offsets_starts[ia];
            const size_t* a_off  = A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];

            const size_t a_rem_rows = aext0 - a_row_off;
            const size_t a_rem_cols = aext1 - a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

            for (size_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const size_t global_i = a_row_off + ii;
                T sum = 0;
                #pragma omp simd reduction(+:sum)
                for (size_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const size_t global_k = a_col_off + kk;

                    sum += A(global_i,global_k) * x(global_k);
                }
                #pragma omp atomic update
                y(global_i) += sum;
            }
        }
    }

    if constexpr (Policy == OpenMPVariant::Simd)
    {

        for (size_t ia = 0; ia < mblocks; ++ia)
        {

            const size_t a_start = A.pooled_offsets_starts[ia];
            const size_t* a_off  = A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];

            const size_t a_rem_rows = aext0 - a_row_off;
            const size_t a_rem_cols = aext1 - a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            for (size_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const size_t global_i = a_row_off + ii;
                T sum = 0;
                #pragma omp simd reduction(+:sum)
                for (size_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const size_t global_k = a_col_off + kk;

                    sum += A(global_i,global_k) * x(global_k);
                }
                y(global_i) += sum;
            }
        }
    }
    else
    {

        for (size_t ia = 0; ia < mblocks; ++ia)
        {

            const size_t a_start = A.pooled_offsets_starts[ia];
            const size_t* a_off  = A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];

            const size_t a_rem_rows = aext0 - a_row_off;
            const size_t a_rem_cols = aext1 - a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

            for (size_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const size_t global_i = a_row_off + ii;
                T sum = 0;
                #pragma omp unroll partial
                for (size_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const size_t global_k = a_col_off + kk;

                    sum += A(global_i,global_k) * x(global_k);
                }
                y(global_i) += sum;
            }
        }
    }
}

#pragma omp end declare target

template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_dot_sparse( const BlockedDataView<T>& A,  const DataBlock<T>& B, DataBlock<T>& C,bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];


    const size_t Cstr0 = C.dpstrides[0];
    const size_t Cstr1 = C.dpstrides[1];

    const size_t aext0 = A.dpextents[0];
    const size_t aext1 = A.dpextents[1];
    const size_t bext0 = B.dpextents[0]; // must equal aext1
    const size_t bext1 = B.dpextents[1];
    if(initialize_to_zero)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp parallel for simd collapse(2)
            for(size_t i=0; i<C.dpextents[0]; i++)
            {
                for(size_t j=0; j<C.dpextents[1]; j++)
                    C.dpdata[i*Cstr0+j*Cstr1]=T(0);
            }
        }
        else if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp simd collapse(2)
            for(size_t i=0; i<C.dpextents[0]; i++)
            {
                for(size_t j=0; j<C.dpextents[1]; j++)
                    C.dpdata[i*Cstr0+j*Cstr1]=T(0);
            }
        }
        else
        {
            for(size_t i=0; i<C.dpextents[0]; i++)
            {
                #pragma omp unroll partial
                for(size_t j=0; j<C.dpextents[1]; j++)
                    C.dpdata[i*Cstr0+j*Cstr1]=T(0);
            }
        }
    }
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for
        for (size_t ia = 0; ia < mblocks; ++ia)
        {
            const size_t a_start = A.pooled_offsets_starts[ia];
            const size_t* a_off  = A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];

            const size_t a_rem_rows = aext0 - a_row_off;
            const size_t a_rem_cols = aext1 - a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            for (size_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const size_t global_i = a_row_off + ii;
                for (size_t jj = 0; jj < bext1; ++jj)
                {
                    T sum = T(0);
                    #pragma omp simd reduction(+:sum)
                    for (size_t kk = 0; kk < a_tile_cols; ++kk)
                    {
                        const size_t global_k = a_col_off + kk;
                        sum += A(global_i,global_k) * B(global_k,jj);
                    }
                    #pragma omp atomic update
                    C(global_i,jj) += sum;
                }
            }
        }
    }

    else  if constexpr (Policy == OpenMPVariant::Simd)
    {
        for (size_t ia = 0; ia < mblocks; ++ia)
        {

            const size_t a_start = A.pooled_offsets_starts[ia];
            const size_t* a_off  = A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];

            const size_t a_rem_rows = aext0 - a_row_off;
            const size_t a_rem_cols = aext1 - a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            for (size_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const size_t global_i = a_row_off + ii;
                for (size_t jj = 0; jj < bext1; ++jj)
                {
                    T sum = T(0);
                    #pragma omp simd reduction(+:sum)
                    for (size_t kk = 0; kk < a_tile_cols; ++kk)
                    {
                        const size_t global_k = a_col_off + kk;
                        sum += A(global_i,global_k) * B(global_k,jj);
                    }
                    C(global_i,jj) += sum;
                }
            }
        }
    }
    else
    {
        for (size_t ia = 0; ia < mblocks; ++ia)
        {

            const size_t a_start = A.pooled_offsets_starts[ia];
            const size_t* a_off  = A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];

            const size_t a_rem_rows = aext0 - a_row_off;
            const size_t a_rem_cols = aext1 - a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            for (size_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const size_t global_i = a_row_off + ii;
                for (size_t jj = 0; jj < bext1; ++jj)
                {
                    T sum = T(0);
                    #pragma omp unroll partial
                    for (size_t kk = 0; kk < a_tile_cols; ++kk)
                    {
                        const size_t global_k = a_col_off + kk;
                        sum += A(global_i,global_k) * B(global_k,jj);
                    }
                    C(global_i,jj) += sum;
                }
            }
        }

    }

}


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_dot_sparse( const BlockedDataView<T>& A, const BlockedDataView<T>& B, DataBlock<T>& C, bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;
    const size_t nblocks = B.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];
    const size_t Bblock_rows = B.block_shape[0];
    const size_t Bblock_cols = B.block_shape[1];

    const size_t Cstr0=C.dpstrides[0];
    const size_t Cstr1=C.dpstrides[1];
    const size_t aext0=A.dpextents[0];
    const size_t aext1=A.dpextents[1];
    const size_t bext0=B.dpextents[0];
    const size_t bext1=B.dpextents[1];


    if(initialize_to_zero)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp parallel for simd collapse(2)
            for(size_t i=0; i<C.dpextents[0]; i++)
            {
                for(size_t j=0; j<C.dpextents[1]; j++)
                    C.dpdata[i*Cstr0+j*Cstr1]=T(0);
            }
        }
        else if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp simd collapse(2)
            for(size_t i=0; i<C.dpextents[0]; i++)
            {
                for(size_t j=0; j<C.dpextents[1]; j++)
                    C.dpdata[i*Cstr0+j*Cstr1]=T(0);
            }
        }
        else
        {
            for(size_t i=0; i<C.dpextents[0]; i++)
            {
                #pragma omp unroll partial
                for(size_t j=0; j<C.dpextents[1]; j++)
                    C.dpdata[i*Cstr0+j*Cstr1]=T(0);
            }
        }
    }
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for collapse(2)
        for (size_t ia = 0; ia < mblocks; ++ia)
        {
            for (size_t jb = 0; jb < nblocks; ++jb)
            {
                const size_t a_start = A.pooled_offsets_starts[ia];

                const size_t* a_off =  A.pooled_offsets_flat + a_start;

                const size_t a_row_off = a_off[0];
                const size_t a_col_off = a_off[1];

                const size_t a_rem_rows = aext0 - a_row_off;
                const size_t a_rem_cols = aext1- a_col_off;

                const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
                const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

                const size_t b_start = B.pooled_offsets_starts[jb];

                const size_t* b_off = B.pooled_offsets_flat + b_start;
                const size_t b_row_off = b_off[0];
                const size_t b_col_off = b_off[1];

                const size_t b_rem_rows =bext0 - b_row_off;
                const size_t b_rem_cols =bext1 - b_col_off;

                const size_t b_tile_rows = (Bblock_rows < b_rem_rows) ? Bblock_rows : b_rem_rows;
                const size_t b_tile_cols = (Bblock_cols < b_rem_cols) ? Bblock_cols : b_rem_cols;

                const size_t a_k_start = a_col_off;
                const size_t a_k_end   = a_col_off + a_tile_cols;

                const size_t b_k_start = b_row_off;
                const size_t b_k_end   = b_row_off + b_tile_rows;

                const size_t k_start = (a_k_start >   b_k_start)  ?   a_k_start:   b_k_start;
                const size_t k_end   = (a_k_end   <   b_k_end)    ?   a_k_end:     b_k_end;

                if (k_start >= k_end)
                {
                    continue;
                }

                for (size_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const size_t global_i = a_row_off + ii;
                    for (size_t jj = 0; jj < b_tile_cols; ++jj)
                    {
                        const  size_t global_j = b_col_off + jj;
                        T sum = T(0);
                        #pragma omp simd reduction(+:sum)
                        for (size_t kk = k_start; kk < k_end; ++kk)
                        {
                            sum += A(global_i,kk)* B(kk,global_j);
                        }
                        #pragma omp atomic update
                        C(global_i, global_j) += sum;
                    }
                }
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {


        for (size_t ia = 0; ia < mblocks; ++ia)
        {
            const size_t a_start = A.pooled_offsets_starts[ia];

            const size_t* a_off =  A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];

            const size_t a_rem_rows = aext0 - a_row_off;
            const size_t a_rem_cols = aext1- a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            const size_t a_k_start = a_col_off;
            const size_t a_k_end   = a_col_off + a_tile_cols;

            for (size_t jb = 0; jb < nblocks; ++jb)
            {

                const size_t b_start = B.pooled_offsets_starts[jb];

                const size_t* b_off = B.pooled_offsets_flat + b_start;
                const size_t b_row_off = b_off[0];
                const size_t b_col_off = b_off[1];

                const size_t b_rem_rows =bext0 - b_row_off;
                const size_t b_rem_cols =bext1 - b_col_off;

                const size_t b_tile_rows = (Bblock_rows < b_rem_rows) ? Bblock_rows : b_rem_rows;
                const size_t b_tile_cols = (Bblock_cols < b_rem_cols) ? Bblock_cols : b_rem_cols;


                const size_t b_k_start = b_row_off;
                const size_t b_k_end   = b_row_off + b_tile_rows;

                const size_t k_start = (a_k_start >   b_k_start)  ?   a_k_start:   b_k_start;
                const size_t k_end   = (a_k_end   <   b_k_end)    ?   a_k_end:     b_k_end;

                if (k_start >= k_end)
                {
                    continue;
                }
                for (size_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const size_t global_i = a_row_off + ii;
                    for (size_t jj = 0; jj < b_tile_cols; ++jj)
                    {
                        const  size_t global_j = b_col_off + jj;
                        T sum = T(0);
                        #pragma omp simd reduction(+:sum)
                        for (size_t kk = k_start; kk < k_end; ++kk)
                        {
                            sum += A(global_i,kk)* B(kk,global_j);
                        }
                        C(global_i, global_j) += sum;
                    }
                }
            }
        }
    }
    else
    {
        for (size_t ia = 0; ia < mblocks; ++ia)
        {
            const size_t a_start = A.pooled_offsets_starts[ia];

            const size_t* a_off =  A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];

            const size_t a_rem_rows = aext0 - a_row_off;
            const size_t a_rem_cols = aext1- a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            const size_t a_k_start = a_col_off;
            const size_t a_k_end   = a_col_off + a_tile_cols;
            for (size_t jb = 0; jb < nblocks; ++jb)
            {

                const size_t b_start = B.pooled_offsets_starts[jb];

                const size_t* b_off = B.pooled_offsets_flat + b_start;
                const size_t b_row_off = b_off[0];
                const size_t b_col_off = b_off[1];

                const size_t b_rem_rows =bext0 - b_row_off;
                const size_t b_rem_cols =bext1 - b_col_off;

                const size_t b_tile_rows = (Bblock_rows < b_rem_rows) ? Bblock_rows : b_rem_rows;
                const size_t b_tile_cols = (Bblock_cols < b_rem_cols) ? Bblock_cols : b_rem_cols;


                const size_t b_k_start = b_row_off;
                const size_t b_k_end   = b_row_off + b_tile_rows;

                const size_t k_start = (a_k_start >   b_k_start)  ?   a_k_start:   b_k_start;
                const size_t k_end   = (a_k_end   <   b_k_end)    ?   a_k_end:     b_k_end;

                if (k_start >= k_end)
                {
                    continue;
                }
                for (size_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const size_t global_i = a_row_off + ii;
                    for (size_t jj = 0; jj < b_tile_cols; ++jj)
                    {
                        const  size_t global_j = b_col_off + jj;
                        T sum = T(0);
                        #pragma omp unroll partial
                        for (size_t kk = k_start; kk < k_end; ++kk)
                        {
                            sum += A(global_i,kk)* B(kk,global_j);
                        }

                        C(global_i, global_j) += sum;
                    }
                }
            }
        }
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_dot_kahan(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = T(0);
                T c=T(0);
                #pragma omp unroll partial
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    T y =  A(i,k) *B(k,j) - c;
                    volatile T t = sum + y;
                    volatile T z = t - sum;
                    c = z - y;
                    sum = t;
                }
                C(i,j)= sum;
            }
        }
    }
    else
    {
        #pragma omp tile sizes(16,16)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = T(0);
                T c=T(0);
                #pragma omp unroll partial
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    T y =  A(i,k) *B(k,j) - c;
                    volatile T t = sum + y;
                    volatile T z = t - sum;
                    c = z - y;
                    sum = t;
                }
                C(i,j)= sum;
            }
        }
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_dot_accumulate_kahan(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = T(0);
                T c=T(0);
                #pragma omp unroll partial
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    T y =  A(i,k) *B(k,j) - c;
                    volatile T t = sum + y;
                    volatile T z = t - sum;
                    c = z - y;
                    sum = t;
                }
                C(i,j)+= sum;
            }
        }
    }
    else     if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp tile sizes(16,16)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = T(0);
                T c=T(0);
                #pragma omp unroll partial
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    T y =  A(i,k) *B(k,j) - c;
                    volatile T t = sum + y;
                    volatile T z = t - sum;
                    c = z - y;
                    sum = t;
                }
                C(i,j)+= sum;
            }
        }
    }


}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::cholesky_decomposition(const DataBlock<T>& A, DataBlock<T>& L, bool initialize_to_zero)
{
    const size_t n = A.dpextents[0];
    L.pconjugate=false;
    if(initialize_to_zero)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                }
            }
        }
        else if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                }
            }
        }
        else
        {
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp unroll partial
                for (size_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                }
            }
        }
    }

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        for (size_t c = 0; c < n; ++c)
        {
            T tmp=T(0);

            #pragma omp  parallel for simd reduction(+:tmp)
            for (size_t k = 0; k < c; ++k)
            {
                const T tmp3=L(c,k);
                tmp+= tmp3 *cond_conj( tmp3);
            }


            tmp=A(c, c)-tmp;
            const T tmp4=sqrt(tmp);
            L(c, c) =tmp4;

            #pragma omp parallel for
            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 =0;
                #pragma omp simd reduction(+:tmp2)
                for (size_t k = 0; k < c; ++k)
                {
                    tmp2 += L(i, k) * cond_conj(L(c, k));
                }
                tmp2= A(i, c)-tmp2;
                L(i, c)=tmp2/tmp4;
            }

        }
    }
    else  if constexpr (Policy == OpenMPVariant::Simd)
    {
        for (size_t c = 0; c < n; ++c)
        {
            T tmp=T(0);

            #pragma omp simd reduction(+:tmp)
            for (size_t k = 0; k < c; ++k)
            {
                const T tmp3=L(c,k);
                tmp+= tmp3 *cond_conj( tmp3);
            }


            tmp=A(c, c)-tmp;
            const T tmp4=sqrt(tmp);
            L(c, c) =tmp4;

            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 =0;
                #pragma omp simd reduction(+:tmp2)
                for (size_t k = 0; k < c; ++k)
                {
                    tmp2 += L(i, k) * cond_conj(L(c, k));
                }
                tmp2= A(i, c)-tmp2;
                L(i, c)=tmp2/tmp4;
            }

        }
    }
    else
    {
        for (size_t c = 0; c < n; ++c)
        {
            T tmp=T(0);

            #pragma omp unroll partial
            for (size_t k = 0; k < c; ++k)
            {
                const T tmp3=L(c,k);
                tmp+= tmp3 *cond_conj( tmp3);
            }


            tmp=A(c, c)-tmp;
            const T tmp4=sqrt(tmp);
            L(c, c) =tmp4;

            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 =0;
                #pragma omp unroll partial
                for (size_t k = 0; k < c; ++k)
                {
                    tmp2 += L(i, k) * cond_conj(L(c, k));
                }
                tmp2= A(i, c)-tmp2;
                L(i, c)=tmp2/tmp4;
            }

        }
    }

}
#pragma omp end declare target





#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::lu_decomposition(const  DataBlock<T>& A, DataBlock<T>& L, DataBlock<T>& U,bool initialize_to_zero)
{

    const size_t n = A.dpextents[0];
    L.pconjugate=false;
    U.pconjugate=false;

    if(initialize_to_zero)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp  parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                    U(i,j)=T(0);
                }
            }
        }
        else if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp  simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                    U(i,j)=T(0);
                }
            }
        }
        else
        {
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp unroll partial
                for (size_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                    U(i,j)=T(0);
                }
            }

        }
    }
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        for (size_t c = 0; c < n; ++c)
        {
            #pragma omp parallel for
            for (size_t i = c; i < n; ++i)
            {
                T temp=T(0);
                #pragma omp  simd reduction(+:temp)
                for (size_t k = 0; k < c; ++k)
                {
                    temp += U( k,i) * L( c,k);
                }
                temp=A(c,i)-temp;
                U(c,i)=temp;
            }

            const T temp4=U(c,c);
            #pragma omp parallel for
            for (size_t i = c; i < n; ++i)
            {
                T temp =T(0);
                #pragma omp simd reduction(+:temp)
                for (size_t k = 0; k < c; ++k)
                {
                    temp += U(k,c) * L( i,k);
                }
                temp=A(i,c)-temp;
                L(i,c)=temp/temp4;
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        for (size_t c = 0; c < n; ++c)
        {
            for (size_t i = c; i < n; ++i)
            {
                T temp=T(0);
                #pragma omp  simd reduction(+:temp)
                for (size_t k = 0; k < c; ++k)
                {
                    temp += U( k,i) * L( c,k);
                }
                temp=A(c,i)-temp;
                U(c,i)=temp;
            }

            const T temp4=U(c,c);
            for (size_t i = c; i < n; ++i)
            {
                T temp =T(0);
                #pragma omp simd reduction(+:temp)
                for (size_t k = 0; k < c; ++k)
                {
                    temp += U(k,c) * L( i,k);
                }
                temp=A(i,c)-temp;
                L(i,c)=temp/temp4;
            }
        }
    }
    else
    {
        for (size_t c = 0; c < n; ++c)
        {
            for (size_t i = c; i < n; ++i)
            {
                T temp=T(0);
                #pragma omp unroll partial
                for (size_t k = 0; k < c; ++k)
                {
                    temp += U( k,i) * L( c,k);
                }
                temp=A(c,i)-temp;
                U(c,i)=temp;
            }

            const T temp4=U(c,c);
            for (size_t i = c; i < n; ++i)
            {
                T temp =T(0);
                #pragma omp unroll partial
                for (size_t k = 0; k < c; ++k)
                {
                    temp += U(k,c) * L( i,k);
                }
                temp=A(i,c)-temp;
                L(i,c)=temp/temp4;
            }
        }
    }

}
#pragma omp end declare target







#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::qr_decomposition( const DataBlock<T>&A, DataBlock<T> Q, DataBlock<T> &R,bool initialize_to_zero, bool with_memmaps)
{
    const size_t n = A.dpextents[0];
    const size_t m = A.dpextents[1];

    Q.pconjugate=false;
    R.pconjugate=false;
    T* tempM;

    if(with_memmaps)
        tempM=Host_Memory_Functions::create_temp_mmap<T>(A.dpdatalength);
    else
        tempM=(T*)omp_alloc(sizeof(T)*A.dpdatalength,omp_default_mem_alloc);


    size_t Mext[2]= {A.dpextents[0],A.dpextents[1]};
    size_t Mstrides[2]= {A.dpstrides[0],A.dpstrides[1]};

    DataBlock<T> M(tempM,A.dpdatalength,A.dprowmajor,A.dprank,Mext,Mstrides,false,-1,false);


    if(initialize_to_zero)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (size_t j = 0; j < n; ++j)
                    Q(i,j) = 0;
                #pragma omp  simd
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                    R(i,j) = 0;
                }
            }
        }
        else  if constexpr (Policy == OpenMPVariant::Simd)
        {
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (size_t j = 0; j < n; ++j)
                    Q(i,j) = 0;
                #pragma omp  simd
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                    R(i,j) = 0;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp unroll partial
                for (size_t j = 0; j < n; ++j)
                    Q(i,j) = 0;
                #pragma omp unroll partial
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                    R(i,j) = 0;
                }
            }
        }
    }
    else
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp  parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                }
            }
        }
        else if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                }
            }
        }
        else
        {
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp unroll partial
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                }
            }
        }
    }

    const size_t pext0=M.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        for (size_t c = 0; c < m; ++c)
        {
            size_t pextv[1];
            size_t pstrv[1];
            DataBlock<T> v = DataBlockUtilities::matrix_column(M,c,pextv,pstrv);
            for (size_t j = 0; j < c; ++j)
            {
                size_t pextu[1];
                size_t pstru[1];

                T dot_pr=T(0);
                DataBlock<T> u = DataBlockUtilities::matrix_column(Q,j,pextu,pstru);
                #pragma omp parallel for simd reduction(+:dot_pr)
                for (size_t i = 0; i < pext0; ++i)
                {
                    dot_pr += cond_conj(u(i)) * v(i);
                }

                const T cdot_pr=dot_pr;
                #pragma omp parallel for simd
                for (size_t i = 0; i < pext0; ++i)
                {
                    v(i) -= cdot_pr * u(i);
                }
            }
            T norm=T(0);
            #pragma omp parallel for simd reduction(+:norm)
            for (size_t i = 0; i < pext0; ++i)
            {
                T val=v(i);
                norm += cond_conj(val) * v(i);
            }

            const T normc= sqrt(norm);
            #pragma omp parallel for simd
            for (size_t i = 0; i < pext0; ++i)
            {
                Q(i,c) = v(i)/normc;
            }
        }

        const size_t rows = Q.dpextents[0];
        const size_t cols = A.dpextents[1];
        const size_t inner_dim = Q.dpextents[1];

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += cond_conj(Q(k,i)) *A(k,j);
                }
                R(i,j)= sum;
            }
        }
    }
    else  if constexpr (Policy == OpenMPVariant::Simd)
    {
        for (size_t c = 0; c < m; ++c)
        {
            size_t pextv[1];
            size_t pstrv[1];
            DataBlock<T> v = M.matrix_column(c,pextv,pstrv);
            for (size_t j = 0; j < c; ++j)
            {
                size_t pextu[1];
                size_t pstru[1];

                T dot_pr=T(0);
                DataBlock<T> u = Q.matrix_column(j,pextu,pstru);
                #pragma omp simd reduction(+:dot_pr)
                for (size_t i = 0; i < pext0; ++i)
                {
                    dot_pr += cond_conj(u(i)) * v(i);
                }

                const T cdot_pr=dot_pr;
                #pragma omp simd
                for (size_t i = 0; i < pext0; ++i)
                {
                    v(i) -= cdot_pr * u(i);
                }
            }
            // Normalize v
            T norm=T(0);
            #pragma omp simd reduction(+:norm)
            for (size_t i = 0; i < pext0; ++i)
            {
                T val=v(i);
                norm += cond_conj(val) * v(i);
            }

            const T normc= sqrt(norm);
            #pragma omp simd
            for (size_t i = 0; i < pext0; ++i)
            {
                Q(i,c) = v(i)/normc;
            }
        }

        const size_t rows = Q.dpextents[0];
        const size_t cols = A.dpextents[1];
        const size_t inner_dim = Q.dpextents[1];

        #pragma omp tile sizes(16, 16)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp  simd reduction(+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += cond_conj(Q(k,i)) *A(k,j);
                }
                R(i,j)= sum;
            }
        }
    }
    else
    {
        for (size_t c = 0; c < m; ++c)
        {
            size_t pextv[1];
            size_t pstrv[1];
            DataBlock<T> v = DataBlockUtilities::matrix_column(M,c,pextv,pstrv);
            for (size_t j = 0; j < c; ++j)
            {
                size_t pextu[1];
                size_t pstru[1];

                T dot_pr=T(0);
                DataBlock<T> u = DataBlockUtilities::matrix_column(Q,j,pextu,pstru);
                #pragma omp unroll partial
                for (size_t i = 0; i < pext0; ++i)
                {
                    dot_pr += cond_conj(u(i)) * v(i);
                }

                const T cdot_pr=dot_pr;
                #pragma omp unroll partial
                for (size_t i = 0; i < pext0; ++i)
                {
                    v(i) -= cdot_pr * u(i);
                }
            }
            T norm=T(0);
            #pragma omp unroll partial
            for (size_t i = 0; i < pext0; ++i)
            {
                T val=v(i);
                norm += cond_conj(val) * v(i);
            }

            const T normc= sqrt(norm);
            #pragma omp unroll partial
            for (size_t i = 0; i < pext0; ++i)
            {
                Q(i,c) = v(i)/normc;
            }
        }

        const size_t rows = Q.dpextents[0];
        const size_t cols = A.dpextents[1];
        const size_t inner_dim = Q.dpextents[1];

        #pragma omp tile sizes(16, 16)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp unroll partial
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += cond_conj(Q(k,i)) *A(k,j);
                }
                R(i,j)= sum;
            }
        }
    }

    if(with_memmaps)
        Host_Memory_Functions::delete_temp_mmap<T>(tempM,A.dpdatalength);
    else
        omp_free(tempM,omp_default_mem_alloc);


}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions::cross_product( const DataBlock<T>& vec1, const  DataBlock<T>& vec2, DataBlock<T>& res)
{
    res(0) = vec1(1) * vec2(2) - vec1(2) * vec2(1);
    res(1) = vec1(2) * vec2(0) - vec1(0) * vec2(2);
    res(2) = vec1(0) * vec2(1) - vec1(1) * vec2(0);

}
#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_dot( const DataBlock<T>& A, const  DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum =T(0);
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k) *B(k,j);
                }
                C(i,j)= sum;
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp tile sizes(16,16)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum =T(0);
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k) *B(k,j);
                }
                C(i,j)= sum;
            }
        }
    }
    else
    {
        #pragma omp tile sizes(16,16)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum =T(0);
                #pragma omp unroll partial
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k) *B(k,j);
                }
                C(i,j)= sum;
            }
        }
    }
}

#pragma omp end declare target





#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_dot_accumulate( const DataBlock<T>& A, const  DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum =T(0);
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k)*B(k,j);
                }
                C(i,j)+= sum;
            }
        }

    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp tile sizes(16,16)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum =T(0);
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k) *B(k,j);
                }
                C(i,j)+= sum;
            }
        }
    }
    else
    {
        #pragma omp tile sizes(16,16)
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                T sum =T(0);
                #pragma omp unroll partial
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k) *B(k,j);
                }
                C(i,j)+= sum;
            }
        }
    }
}
#pragma omp end declare target











#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_add(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j <m ; ++j)
            {
                C(i,j) =A(i,j)+B(i,j);
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j <m ; ++j)
            {
                C(i,j) =A(i,j)+B(i,j);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < n; ++i)
        {
            #pragma omp unroll partial
            for (size_t j = 0; j <m ; ++j)
            {
                C(i,j) =A(i,j)+B(i,j);
            }
        }
    }


}
#pragma omp end declare target







#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_add_accumulate( DataBlock<T>& A,const DataBlock<T>& B)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j <m ; ++j)
            {
                A(i,j)+=B(i,j);
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j <m ; ++j)
            {
                A(i,j)+=B(i,j);
            }
        }

    }
    else
    {
        for (size_t i = 0; i < n; ++i)
        {
            #pragma omp unroll partial
            for (size_t j = 0; j <m ; ++j)
            {
                A(i,j)+=B(i,j);
            }
        }
    }


}
#pragma omp end declare target





#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_subtract(const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for simd collapse(2)
        for (size_t i = 0; i <n; ++i)
        {
            for (size_t j = 0; j < m; ++j)
            {
                C(i,j) =A(i,j)-B(i,j);
            }
        }
    }
    else  if constexpr (Policy == OpenMPVariant::Simd)
    {

        #pragma omp simd collapse(2)
        for (size_t i = 0; i <n; ++i)
        {
            for (size_t j = 0; j < m; ++j)
            {
                C(i,j) =A(i,j)-B(i,j);
            }
        }
    }
    else
    {
        for (size_t i = 0; i <n; ++i)
        {
            #pragma omp unroll partial
            for (size_t j = 0; j < m; ++j)
            {
                C(i,j) =A(i,j)-B(i,j);
            }
        }
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_subtract_accumulate( DataBlock<T>& A,const  DataBlock<T>& B)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for (size_t i = 0; i <n; ++i)
        {
            for (size_t j = 0; j < m; ++j)
            {
                A(i,j)-=B(i,j);
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for (size_t i = 0; i <n; ++i)
        {
            for (size_t j = 0; j < m; ++j)
            {
                A(i,j)-=B(i,j);
            }
        }
    }
    else
    {
        for (size_t i = 0; i <n; ++i)
        {
            #pragma omp unroll partial
            for (size_t j = 0; j < m; ++j)
            {
                A(i,j)-=B(i,j);
            }
        }
    }


}
#pragma omp end declare target





#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_vector( const DataBlock<T>&M,const  DataBlock<T>& V, DataBlock<T>& C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for
        for (size_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j <  m; ++j)
            {
                sum+= M(i, j) * V(j);
            }
            C(i)=sum;
        }

    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {

        for (size_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j <  m; ++j)
            {
                sum+= M(i, j) * V(j);
            }
            C(i)=sum;
        }

    }
    else
    {

        for (size_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            #pragma omp unroll partial
            for (size_t j = 0; j <  m; ++j)
            {
                sum+= M(i, j) * V(j);
            }
            C(i)=sum;
        }
    }


}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_vector_kahan( const DataBlock<T>&M,const  DataBlock<T>& V, DataBlock<T>& C)
{

    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for
        for (size_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            T c=T(0);
            #pragma omp unroll partial
            for (size_t j = 0; j <  m; ++j)
            {
                T y = M(i, j) * V(j) - c;
                volatile T t = sum + y;
                volatile T z = t - sum;
                c = z - y;
                sum = t;
            }
            C(i)=sum;
        }
    }
    else
    {

        for (size_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            T c=T(0);
            #pragma omp unroll partial
            for (size_t j = 0; j <  m; ++j)
            {
                T y = M(i, j) * V(j) - c;
                volatile T t = sum + y;
                volatile T z = t - sum;
                c = z - y;
                sum = t;
            }
            C(i)=sum;
        }
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_vector_kahan( const DataBlock<T>&M,const T* V, DataBlock<T>& C)
{

    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for
        for (size_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            T c=T(0);
            #pragma omp unroll partial
            for (size_t j = 0; j <  m; ++j)
            {
                T y = M(i, j) * V[j] - c;
                volatile T t = sum + y;
                volatile T z = t - sum;
                c = z - y;
                sum = t;
            }
            C(i)=sum;
        }
    }
    else
    {
        for (size_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            T c=T(0);
            #pragma omp unroll partial
            for (size_t j = 0; j <  m; ++j)
            {
                T y = M(i, j) * V[j] - c;
                volatile T t = sum + y;
                volatile T z = t - sum;
                c = z - y;
                sum = t;
            }
            C(i)=sum;
        }
    }
}
#pragma omp end declare target



#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_vector( const DataBlock<T>&M, const T*V, DataBlock<T> & C)
{
    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for
        for (size_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j <  m; ++j)
            {
                sum+= M(i, j) * V[j];
            }
            C(i)=sum;
        }
    }
    if constexpr (Policy == OpenMPVariant::Simd)
    {
        for (size_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            #pragma omp  simd reduction(+:sum)
            for (size_t j = 0; j <  m; ++j)
            {
                sum+= M(i, j) * V[j];
            }
            C(i)=sum;
        }
    }
    else
    {
        for (size_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            #pragma omp unroll partial
            for (size_t j = 0; j <  m; ++j)
            {
                sum+= M(i, j) * V[j];
            }
            C(i)=sum;
        }
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::vector_add( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res)
{
    const size_t n=vec1.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; ++i)
        {
            res(i)= vec1(i)+vec2(i);
        }

    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i)
        {
            res(i) = vec1(i)+vec2(i);
        }
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < n; ++i)
        {
            res(i)= vec1(i)+vec2(i);
        }
    }

}
#pragma omp end declare target



#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::vector_add_accumulate( DataBlock<T>& vec1,const  DataBlock<T>& vec2)
{
    const size_t n=vec1.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; ++i)
        {
            vec1(i)+=vec2(i);
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i)
        {
            vec1(i)+=vec2(i);
        }
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < n; ++i)
        {
            vec1(i)+=vec2(i);
        }
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::vector_subtract( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res)
{
    const size_t n=vec1.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; ++i)
        {
            res(i) = vec1(i)-vec2(i);
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {

        #pragma omp simd
        for (size_t i = 0; i < n; ++i)
        {
            res(i) = vec1(i)-vec2(i);
        }
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < n; ++i)
        {
            res(i) = vec1(i)-vec2(i);
        }
    }

}
#pragma omp end declare target





#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
T In_Kernel_Mathfunctions::dot_product(const  DataBlock<T> &vec1, const DataBlock<T> &vec2)
{
    const size_t n=vec1.dpextents[0];
    T result = T(0);
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for reduction(+:result)
        for (size_t i = 0; i < n; ++i)
        {
            result += cond_conj( vec1(i)) * vec2(i);
        }
        return result;
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {

        #pragma omp simd reduction(+:result)
        for (size_t i = 0; i < n; ++i)
        {
            result += cond_conj( vec1(i)) * vec2(i);
        }
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < n; ++i)
        {
            result += cond_conj( vec1(i)) * vec2(i);
        }
    }

    return result;
}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
T In_Kernel_Mathfunctions::dot_product_kahan(const DataBlock<T> &vec1, const DataBlock<T> &vec2)
{
    const size_t n=vec1.dpextents[0];
    T result = T(0);
    T c_final = T(0);

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        const int total_threads = omp_get_max_threads();
        if (n < (size_t)total_threads)
        {
            #pragma omp unroll partial
            for (int i = 0; i < n; ++i)
            {
                T y = cond_conj( vec1(i))* vec2(i)- c_final;
                volatile T t = result + y;
                volatile T z = t - result;
                c_final=z-y;
                result = t;
            }
        }
        else
        {
            constexpr int MAX_STATIC_THREADS = 256;
            T thread_sums[MAX_STATIC_THREADS];
            T thread_cs[MAX_STATIC_THREADS];
            const int actual_workers = (total_threads > MAX_STATIC_THREADS) ? MAX_STATIC_THREADS : total_threads;

            #pragma omp parallel for simd
            for (int idx = 0; idx < actual_workers; ++idx)
            {
                thread_sums[idx] = T(0);
                thread_cs[idx] = T(0);
            }

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                if (tid < actual_workers)
                {
                    T local_sum = T(0);
                    T c = T(0);
                    #pragma omp unroll partial
                    for (size_t i = tid; i < n; i += actual_workers)
                    {
                        T term= cond_conj( vec1(i)) * vec2(i);
                        T y = term - c;
                        volatile T t = local_sum + y;
                        volatile T z = t - local_sum;
                        c = z - y;
                        local_sum = t;
                    }

                    thread_sums[tid] = local_sum;
                    thread_cs[tid]   = c;
                }
            }

            #pragma omp unroll partial
            for (int tid = 0; tid < actual_workers; ++tid)
            {

                T y1 = thread_sums[tid] - c_final;
                volatile T t1 = result + y1;
                volatile T z1 = t1 - result;
                c_final = z1 - y1;
                result = t1;


                T y2 = thread_cs[tid] - c_final;
                volatile T t2 = result + y2;
                volatile T z2 = t2 - result;
                c_final = z2 - y2;
                result = t2;
            }
        }
    }
    else
    {
        #pragma omp unroll partial
        for (int i = 0; i < n; ++i)
        {
            T y = cond_conj( vec1(i))* vec2(i)- c_final;
            volatile T t = result + y;
            volatile T z = t - result;
            c_final=z-y;
            result = t;
        }
    }
    return result;
}
#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_scalar(  const DataBlock<T>& M, const T V, DataBlock<T>& C)
{

    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for (size_t i = 0; i <n; ++i)
        {
            for (size_t j = 0; j <  m; ++j)
            {
                C(i,j)= M(i, j) * V;
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for (size_t i = 0; i <n; ++i)
        {
            for (size_t j = 0; j <  m; ++j)
            {
                C(i,j)= M(i, j) * V;
            }
        }
    }
    else
    {

        for (size_t i = 0; i <n; ++i)
        {
            #pragma omp unroll partial
            for (size_t j = 0; j <  m; ++j)
            {
                C(i,j)= M(i, j) * V;
            }
        }
    }

}
#pragma omp end declare target





#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_scalar_accumulate(   DataBlock<T>& M, const T V)
{

    const size_t n=M.dpextents[0];
    const size_t m= M.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for (size_t i = 0; i <n; ++i)
        {
            for (size_t j = 0; j <  m; ++j)
            {
                M(i, j) *= V;
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for (size_t i = 0; i <n; ++i)
        {
            for (size_t j = 0; j <  m; ++j)
            {
                M(i, j) *= V;
            }
        }
    }
    else
    {
        for (size_t i = 0; i <n; ++i)
        {
            #pragma omp unroll partial
            for (size_t j = 0; j <  m; ++j)
            {
                M(i, j) *= V;
            }
        }

    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::vector_multiply_scalar( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res)
{

    const size_t n=vec.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; ++i)
        {
            res(i) = vec(i)*scalar;
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i)
        {
            res(i) = vec(i)*scalar;
        }
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < n; ++i)
        {
            res(i) = vec(i)*scalar;
        }
    }
}

#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::vector_multiply_scalar_accumulate( DataBlock<T>& vec,const T scalar)
{
    const size_t n=vec.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; ++i)
        {
            vec(i)*=scalar;
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i)
        {
            vec(i)*=scalar;
        }
    }
    else
    {
        #pragma omp unroll partial
        for (size_t i = 0; i < n; ++i)
        {
            vec(i)*=scalar;
        }
    }
}

#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
T  In_Kernel_Mathfunctions::kahan_sum(const T *arr, size_t n)
{
    T sum = T(0);
    T c = T(0);
    #pragma omp unroll partial
    for (size_t i = 0; i < n; ++i)
    {
        T y = arr[i] - c;
        volatile T t = sum + y;
        volatile T z=t-sum;
        c = z - y;
        sum = t;
    }
    return sum;
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
T In_Kernel_Mathfunctions::neumaier_sum(const T* arr, size_t n)
{

    if constexpr (is_complex<T>())
    {
        using ValueType = typename T::value_type;

        ValueType r_sum = ValueType(0);
        ValueType r_comp = ValueType(0);
        ValueType i_sum = ValueType(0);
        ValueType i_comp = ValueType(0);
        for (size_t i = 0; i < n; ++i)
        {

            ValueType rx = arr[i].real();
            volatile ValueType rt = r_sum + rx;

            ValueType z1 = (r_sum <  ValueType(0)) ? -r_sum : r_sum;
            ValueType z2 = (rx <  ValueType(0)) ? -rx : rx;
            if (z1 >=z2)
            {
                volatile ValueType rz = (r_sum - rt) + rx;
                r_comp += rz;
            }
            else
            {
                volatile ValueType rz = (rx - rt) + r_sum;
                r_comp += rz;
            }
            r_sum = rt;

            ValueType ix = arr[i].imag();
            volatile ValueType it = i_sum + ix;

            z1 = (i_sum <  ValueType(0)) ? -i_sum : i_sum;
            z2 = (ix <  ValueType(0)) ? -ix : ix;
            if (z1>= z2)
            {
                volatile ValueType iz = (i_sum - it) + ix;
                i_comp += iz;
            }
            else
            {
                volatile ValueType iz = (ix - it) + i_sum;
                i_comp += iz;
            }
            i_sum = it;
        }
        return T(r_sum + r_comp, i_sum + i_comp);
    }
    else
    {
        T sum = T(0);
        T comp = T(0);
        for (size_t i = 0; i < n; ++i)
        {
            T x = arr[i];
            volatile T t = sum + x;

            T z1 = (sum < T(0)) ? -sum : sum;
            T z2 = (x < T(0)) ? -x : x;

            if (z1 >= z2)
            {
                volatile T z = (sum - t) + x;
                comp += z;
            }
            else
            {
                volatile T z = (x - t) + sum;
                comp += z;
            }
            sum = t;
        }
        return sum + comp;
    }
}
#pragma omp end declare target
#endif
