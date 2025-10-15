#ifndef INKERNELMATHFUNCTIONS
#define INKERNELMATHFUNCTIONS

#include "cmath"
#include "datablock.h"
#include "datablockcontainer.h"

using namespace std;



#pragma omp begin declare target
template <typename T>
class In_Kernel_Mathfunctions
{
public:
    inline static void cholesky_decomposition_w(const DataBlock<T>& A, DataBlock<T>& L,bool initialize_to_zero=true);
    inline static void lu_decomposition_w(const  DataBlock<T>& dA, DataBlock<T>& dL, DataBlock<T>& dU,bool initialize_to_zero=true);
    inline static void qr_decomposition_w( const DataBlock<T>&A, DataBlock<T> Q, DataBlock<T> &R,bool initialize_to_zero=true,bool with_memmaps=false);

    inline static void cross_product( const DataBlock<T>& vec1,const   DataBlock<T>& vec2, DataBlock<T>& res);

    inline static void matrix_multiply_dot_w( const DataBlock<T>& A,  const DataBlock<T>& B, DataBlock<T>& C);
    inline static void matrix_multiply_dot_w_kahan( const DataBlock<T>& A, const  DataBlock<T>& B, DataBlock<T>& C);
    inline static void matrix_multiply_dot_v( const DataBlock<T>& A, const  DataBlock<T>& B, DataBlock<T>& C);
    inline static void matrix_multiply_dot_s(const  DataBlock<T>& A, const  DataBlock<T>& B, DataBlock<T>& C);
    inline static void matrix_multiply_dot_kahan_w(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C);
    inline static void matrix_multiply_dot_kahan_s(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C);

    inline static void matrix_add_w(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C);
    inline static void matrix_add_v(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C);
    inline static void matrix_add_s(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C);

    inline static void matrix_subtract_w(const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C);
    inline static void matrix_subtract_v(const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C);
    inline static void matrix_subtract_s(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C);

    inline static void matrix_multiply_vector_w( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T> &C);
    inline static void matrix_multiply_vector_v( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T> &C);
    inline static void matrix_multiply_vector_s( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T> &C);
    inline static void matrix_multiply_vector_kahan_s( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T>& C);
    inline static void matrix_multiply_vector_kahan_w( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T> &C);

    inline static void matrix_multiply_vector_s( const DataBlock<T>&M,const  T*V, DataBlock<T> & C);
    inline static void matrix_multiply_vector_v( const DataBlock<T>&M,const  T*V, DataBlock<T> & C);
    inline static void matrix_multiply_vector_w( const DataBlock<T>&M,const  T*V, DataBlock<T> & C);
    inline static void matrix_multiply_vector_kahan_w( const DataBlock<T>&M,const  T*V, DataBlock<T> & C);
    inline static void matrix_multiply_vector_kahan_s( const DataBlock<T>&M,const  T*V, DataBlock<T> & C);

    inline static void vector_add_s(const  DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res);
    inline static void vector_add_v( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res);
    inline static void vector_add_w( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res);

    inline static void vector_subtract_w(const  DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res);
    inline static void vector_subtract_v( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res);
    inline static void vector_subtract_s( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res);

    inline static T dot_product_s( const DataBlock<T> &vec1,const  DataBlock<T> &vec2);
    inline static T dot_product_v( const DataBlock<T> &vec1, const DataBlock<T> &vec2);
    inline static T dot_product_w( const DataBlock<T> &vec1,const  DataBlock<T> &vec2);
    inline static T dot_product_w_kahan(const  DataBlock<T> &vec1, const DataBlock<T> &vec2);

    inline static void matrix_multiply_scalar_s( const  DataBlock<T>& M, const T V, DataBlock<T>& C);
    inline static void matrix_multiply_scalar_v( const  DataBlock<T>& M,const  T V, DataBlock<T>& C);
    inline static void matrix_multiply_scalar_w( const  DataBlock<T>& M,const  T V, DataBlock<T>& C);

    inline static void vector_multiply_scalar_s( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res);
    inline static void vector_multiply_scalar_v( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res);
    inline static void vector_multiply_scalar_w( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res);

    inline static T  kahan_sum(const T *arr,size_t n);
    inline static T  neumaier_sum(const T*arr,size_t n);

    inline static void matrix_multiply_dot_sparse_w(const BlockedDataView<T>& Ablocks, const BlockedDataView<T>& Bblocks, DataBlock<T>& C,bool initialize_output_to_zero=true);
    inline static void matrix_multiply_dot_sparse_v(const BlockedDataView<T>& Ablocks, const BlockedDataView<T>& Bblocks, DataBlock<T>& C,bool initialize_output_to_zero=true);
    inline static void matrix_multiply_dot_sparse_s(const BlockedDataView<T>& Ablocks, const BlockedDataView<T>& Bblocks, DataBlock<T>& C,bool initialize_output_to_zero=true);

    inline static void matrix_multiply_dot_sparse_w(const BlockedDataView<T>& Ablocks, const DataBlock<T>& Bblocks, DataBlock<T>& C,bool initialize_output_to_zero=true);
    inline static void matrix_multiply_dot_sparse_v(const BlockedDataView<T>& Ablocks, const DataBlock<T>& Bblocks, DataBlock<T>& C,bool initialize_output_to_zero=true);
    inline static void matrix_multiply_dot_sparse_s(const BlockedDataView<T>& Ablocks, const DataBlock<T>& Bblocks, DataBlock<T>& C,bool initialize_output_to_zero=true);


    inline static  void matrix_vector_multiply_sparse_s(const BlockedDataView<T>& A, const DataBlock<T>& x,  DataBlock<T>& y,bool initialize_output_to_zero=true);
    inline static  void matrix_vector_multiply_sparse_v(const BlockedDataView<T>& A, const DataBlock<T>& x,  DataBlock<T>& y,bool initialize_output_to_zero=true);
    inline static  void matrix_vector_multiply_sparse_w(const BlockedDataView<T>& A, const DataBlock<T>& x,  DataBlock<T>& y,bool initialize_output_to_zero=true);

    inline static  void matrix_vector_multiply_sparse_s(const BlockedDataView<T>& A, const BlockedDataView<T>& x, DataBlock<T>& y,bool initialize_output_to_zero=true);
    inline static  void matrix_vector_multiply_sparse_v(const BlockedDataView<T>& A, const BlockedDataView<T>& x, DataBlock<T>& y,bool initialize_output_to_zero=true);
    inline static  void matrix_vector_multiply_sparse_w(const BlockedDataView<T>& A, const BlockedDataView<T>& x, DataBlock<T>& y,bool initialize_output_to_zero=true);
};
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_vector_multiply_sparse_w(  const BlockedDataView<T>& A, const BlockedDataView<T>& x,  DataBlock<T>& y,bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;
    const size_t nblocks = x.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];
    const size_t Xblock_size = x.block_shape[0];

    const size_t Astr0 = A.dblock.dpstrides[0];
    const size_t Astr1 = A.dblock.dpstrides[1];
    const size_t Xstr0 = x.dblock.dpstrides[0];

    const size_t aext0 = A.dblock.dpextents[0];
    const size_t aext1 = A.dblock.dpextents[1];
    const size_t xext  = x.dblock.dpextents[0];

    const size_t ystr0 = y.dpstrides[0];

    if(initialize_to_zero)
    {
        #pragma omp parallel for simd shared(y)
        for(size_t i=0; i<y.dpextents[0]; i++)
            y.dpdata[i*ystr0]=T(0);
    }


    #pragma omp parallel for collapse(2) shared(A,x,y)
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

            // overlap check in "k" dimension
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
                    const size_t a_index = global_i * Astr0 + kk * Astr1;
                    const size_t x_index = kk * Xstr0;
                    sum += A.dblock.dpdata[a_index] * x.dblock.dpdata[x_index];
                }
                #pragma omp atomic update
                y.dpdata[global_i * ystr0] += sum;
            }
        }
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_vector_multiply_sparse_v(
    const BlockedDataView<T>& A,  const BlockedDataView<T>& x,  DataBlock<T>& y,bool initialize_to_zero )
{
    const size_t mblocks = A.usedblocks;
    const size_t nblocks = x.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];
    const size_t Xblock_size = x.block_shape[0];

    const size_t Astr0 = A.dblock.dpstrides[0];
    const size_t Astr1 = A.dblock.dpstrides[1];
    const size_t Xstr0 = x.dblock.dpstrides[0];

    const size_t aext0 = A.dblock.dpextents[0];
    const size_t aext1 = A.dblock.dpextents[1];
    const size_t xext  = x.dblock.dpextents[0];

    const size_t ystr0 = y.dpstrides[0];
    if(initialize_to_zero)
    {
        #pragma omp simd
        for(size_t i=0; i<y.dpextents[0]; i++)
            y.dpdata[i*ystr0]=T(0);
    }


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

        for (size_t jb = 0; jb < nblocks; ++jb)
        {
            const size_t x_start = x.pooled_offsets_starts[jb];
            const size_t* x_off  = x.pooled_offsets_flat + x_start;

            const size_t x_off0 = x_off[0];
            const size_t x_rem  = xext - x_off0;
            const size_t x_tile = (Xblock_size < x_rem) ? Xblock_size : x_rem;

            // overlap check in "k" dimension
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
                    const size_t a_index = global_i * Astr0 + kk * Astr1;
                    const size_t x_index = kk * Xstr0;
                    sum += A.dblock.dpdata[a_index] * x.dblock.dpdata[x_index];
                }
                y.dpdata[global_i * ystr0] +=sum;
            }
        }
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_vector_multiply_sparse_s(
    const BlockedDataView<T>& A,    const BlockedDataView<T>& x,    DataBlock<T>& y,bool initialize_to_zero )
{
    const size_t mblocks = A.usedblocks;
    const size_t nblocks = x.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];
    const size_t Xblock_size = x.block_shape[0];

    const size_t Astr0 = A.dblock.dpstrides[0];
    const size_t Astr1 = A.dblock.dpstrides[1];
    const size_t Xstr0 = x.dblock.dpstrides[0];

    const size_t aext0 = A.dblock.dpextents[0];
    const size_t aext1 = A.dblock.dpextents[1];
    const size_t xext  = x.dblock.dpextents[0];

    const size_t ystr0 = y.dpstrides[0];

    if(initialize_to_zero)
    {
        for(size_t i=0; i<y.dpextents[0]; i++)
            y.dpdata[i*ystr0]=T(0);
    }


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

        for (size_t jb = 0; jb < nblocks; ++jb)
        {
            const size_t x_start = x.pooled_offsets_starts[jb];
            const size_t* x_off  = x.pooled_offsets_flat + x_start;

            const size_t x_off0 = x_off[0];
            const size_t x_rem  = xext - x_off0;
            const size_t x_tile = (Xblock_size < x_rem) ? Xblock_size : x_rem;

            // overlap check in "k" dimension
            const size_t k_start = (a_col_off> x_off0) ? a_col_off:x_off0;
            const size_t a= a_col_off + a_tile_cols;
            const size_t b=x_off0 + x_tile;
            const size_t k_end   =(a<b)?a:b;

            if (k_start >= k_end) continue;

            for (size_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const size_t global_i = a_row_off + ii;
                T sum = 0;

                for (size_t kk = k_start; kk < k_end; ++kk)
                {
                    const size_t a_index = global_i * Astr0 + kk * Astr1;
                    const size_t x_index = kk * Xstr0;
                    sum += A.dblock.dpdata[a_index] * x.dblock.dpdata[x_index];
                }
                y.dpdata[global_i * ystr0]+= sum;
            }
        }
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_vector_multiply_sparse_w( const BlockedDataView<T>& A, const DataBlock<T>& x, DataBlock<T>& y,bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];

    const size_t Astr0 = A.dblock.dpstrides[0];
    const size_t Astr1 = A.dblock.dpstrides[1];
    const size_t Xstr0 = x.dpstrides[0];

    const size_t aext0 = A.dblock.dpextents[0];
    const size_t aext1 = A.dblock.dpextents[1];

    const size_t ystr0 = y.dpstrides[0];

    if(initialize_to_zero)
    {
        #pragma omp parallel for simd shared(y)
        for(size_t i=0; i<y.dpextents[0]; i++)
            y.dpdata[i*ystr0]=T(0);
    }



    #pragma omp parallel for shared(A,x,y)
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
                const size_t a_index  = global_i * Astr0 + global_k * Astr1;
                const size_t x_index  = global_k * Xstr0;
                sum += A.dblock.dpdata[a_index] * x.dpdata[x_index];
            }
            #pragma omp atomic update
            y.dpdata[global_i * ystr0] += sum;
        }
    }
}
#pragma omp end declare target


#pragma omp begin declare target

template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_vector_multiply_sparse_v(const BlockedDataView<T>& A,    const DataBlock<T>& x,     DataBlock<T>& y,bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];

    const size_t Astr0 = A.dblock.dpstrides[0];
    const size_t Astr1 = A.dblock.dpstrides[1];
    const size_t Xstr0 = x.dpstrides[0];

    const size_t aext0 = A.dblock.dpextents[0];
    const size_t aext1 = A.dblock.dpextents[1];

    const size_t ystr0 = y.dpstrides[0];

    if(initialize_to_zero)
    {
        #pragma omp simd
        for(size_t i=0; i<y.dpextents[0]; i++)
            y.dpdata[i*ystr0]=T(0);
    }

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
            T sum =T(0);  // accumulate
            #pragma omp simd reduction(+:sum)
            for (size_t kk = 0; kk < a_tile_cols; ++kk)
            {
                const size_t global_k = a_col_off + kk;
                const size_t a_index  = global_i * Astr0 + global_k * Astr1;
                const size_t x_index  = global_k * Xstr0;
                sum += A.dblock.dpdata[a_index] * x.dpdata[x_index];
            }

            y.dpdata[global_i * ystr0] += sum;
        }
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_vector_multiply_sparse_s(  const BlockedDataView<T>& A,  const DataBlock<T>& x, DataBlock<T>& y, bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];

    const size_t Astr0 = A.dblock.dpstrides[0];
    const size_t Astr1 = A.dblock.dpstrides[1];
    const size_t Xstr0 = x.dpstrides[0];

    const size_t aext0 = A.dblock.dpextents[0];
    const size_t aext1 = A.dblock.dpextents[1];

    const size_t ystr0 = y.dpstrides[0];
    if(initialize_to_zero)
    {
        for(size_t i=0; i<y.dpextents[0]; i++)
            y.dpdata[i*ystr0]=T(0);
    }

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
            T sum =T(0) ;
            for (size_t kk = 0; kk < a_tile_cols; ++kk)
            {
                const size_t global_k = a_col_off + kk;
                const size_t a_index  = global_i * Astr0 + global_k * Astr1;
                const size_t x_index  = global_k * Xstr0;
                sum += A.dblock.dpdata[a_index] * x.dpdata[x_index];
            }

            y.dpdata[global_i * ystr0] += sum;
        }
    }
}
#pragma omp end declare target

template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_sparse_w( const BlockedDataView<T>& A,  const DataBlock<T>& B, DataBlock<T>& C,bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];

    const size_t Astr0 = A.dblock.dpstrides[0];
    const size_t Astr1 = A.dblock.dpstrides[1];
    const size_t Bstr0 = B.dpstrides[0];
    const size_t Bstr1 = B.dpstrides[1];
    const size_t Cstr0 = C.dpstrides[0];
    const size_t Cstr1 = C.dpstrides[1];

    const size_t aext0 = A.dblock.dpextents[0];
    const size_t aext1 = A.dblock.dpextents[1];
    const size_t bext0 = B.dpextents[0]; // must equal aext1
    const size_t bext1 = B.dpextents[1];
    if(initialize_to_zero)
    {
        #pragma omp parallel for simd collapse(2) shared(C)
        for(size_t i=0; i<C.dpextents[0]; i++)
            for(size_t j=0; j<C.dpextents[1]; j++)
                C.dpdata[i*Cstr0+j*Cstr1]=T(0);
    }
    #pragma omp parallel for shared(A,B,C)
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

            for (size_t jj = 0; jj < bext1; ++jj)  // loop over all columns of B
            {
                T sum = T(0);
                #pragma omp simd reduction(+:sum)
                for (size_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const size_t global_k = a_col_off + kk;

                    const size_t a_index = global_i * Astr0 + global_k * Astr1;
                    const size_t b_index = global_k * Bstr0 + jj * Bstr1;

                    sum += A.dblock.dpdata[a_index] * B.dpdata[b_index];
                }
                #pragma omp atomic update
                C.dpdata[global_i * Cstr0 + jj * Cstr1] += sum;
            }
        }
    }
}


template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_sparse_v( const BlockedDataView<T>& A,  const DataBlock<T>& B, DataBlock<T>& C,bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];

    const size_t Astr0 = A.dblock.dpstrides[0];
    const size_t Astr1 = A.dblock.dpstrides[1];
    const size_t Bstr0 = B.dpstrides[0];
    const size_t Bstr1 = B.dpstrides[1];
    const size_t Cstr0 = C.dpstrides[0];
    const size_t Cstr1 = C.dpstrides[1];

    const size_t aext0 = A.dblock.dpextents[0];
    const size_t aext1 = A.dblock.dpextents[1];
    const size_t bext0 = B.dpextents[0]; // must equal aext1
    const size_t bext1 = B.dpextents[1];
    if(initialize_to_zero)
    {
        #pragma omp simd collapse(2)
        for(size_t i=0; i<C.dpextents[0]; i++)
            for(size_t j=0; j<C.dpextents[1]; j++)
                C.dpdata[i*Cstr0+j*Cstr1]=T(0);
    }
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

        // Multiply this sparse tile of A with the corresponding slice of dense B
        for (size_t ii = 0; ii < a_tile_rows; ++ii)
        {
            const size_t global_i = a_row_off + ii;

            for (size_t jj = 0; jj < bext1; ++jj)  // loop over all columns of B
            {
                T sum = T(0);
                #pragma omp simd reduction(+:sum)
                for (size_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const size_t global_k = a_col_off + kk;

                    const size_t a_index = global_i * Astr0 + global_k * Astr1;
                    const size_t b_index = global_k * Bstr0 + jj * Bstr1;

                    sum += A.dblock.dpdata[a_index] * B.dpdata[b_index];
                }

                C.dpdata[global_i * Cstr0 + jj * Cstr1] += sum;
            }
        }
    }
}


template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_sparse_s( const BlockedDataView<T>& A,  const DataBlock<T>& B, DataBlock<T>& C,bool initialize_to_zero)
{
    const size_t mblocks = A.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];

    const size_t Astr0 = A.dblock.dpstrides[0];
    const size_t Astr1 = A.dblock.dpstrides[1];
    const size_t Bstr0 = B.dpstrides[0];
    const size_t Bstr1 = B.dpstrides[1];
    const size_t Cstr0 = C.dpstrides[0];
    const size_t Cstr1 = C.dpstrides[1];

    const size_t aext0 = A.dblock.dpextents[0];
    const size_t aext1 = A.dblock.dpextents[1];
    const size_t bext0 = B.dpextents[0]; // must equal aext1
    const size_t bext1 = B.dpextents[1];
    if(initialize_to_zero)
    {
        for(size_t i=0; i<C.dpextents[0]; i++)
            for(size_t j=0; j<C.dpextents[1]; j++)
                C.dpdata[i*Cstr0+j*Cstr1]=T(0);
    }
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

                for (size_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const size_t global_k = a_col_off + kk;

                    const size_t a_index = global_i * Astr0 + global_k * Astr1;
                    const size_t b_index = global_k * Bstr0 + jj * Bstr1;

                    sum += A.dblock.dpdata[a_index] * B.dpdata[b_index];
                }
                C.dpdata[global_i * Cstr0 + jj * Cstr1] += sum;
            }
        }
    }
}



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_sparse_w( const BlockedDataView<T>& A, const BlockedDataView<T>& B, DataBlock<T>& C, bool initialize_to_zero)
{
    // both A and B are assumed 2D
    const size_t mblocks = A.usedblocks;
    const size_t nblocks = B.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];
    const size_t Bblock_rows = B.block_shape[0];
    const size_t Bblock_cols = B.block_shape[1];

    const size_t str0=C.dpstrides[0];
    const size_t str1=C.dpstrides[1];
    const size_t aext0=A.dblock.dpextents[0];
    const size_t aext1=A.dblock.dpextents[1];
    const size_t bext0=B.dblock.dpextents[0];
    const size_t bext1=B.dblock.dpextents[1];

    const size_t Astr0=A.dblock.dpstrides[0];
    const size_t Astr1=A.dblock.dpstrides[1];
    const size_t Bstr0=B.dblock.dpstrides[0];
    const size_t Bstr1=B.dblock.dpstrides[1];
    if(initialize_to_zero)
    {
        #pragma omp parallel for simd collapse(2)
        for(size_t i=0; i<C.dpextents[0]; i++)
            for(size_t j=0; j<C.dpextents[1]; j++)
                C.dpdata[i*str0+j*str1]=T(0);
    }
    #pragma omp parallel for collapse(2) shared(A,B,C)
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
                        const size_t a_index = (global_i *Astr0) +  (kk * Astr1);
                        const size_t b_index = (kk * Bstr0) +        (global_j * Bstr1);

                        sum += A.dblock.dpdata[a_index] * B.dblock.dpdata[b_index];
                    }
                    #pragma omp atomic update
                    C.dpdata[global_i*str0+ global_j*str1] += sum;
                }
            }
        }
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_sparse_v(  const BlockedDataView<T>& A, const BlockedDataView<T>& B,  DataBlock<T>& C,bool initialize_to_zero)
{
    // both A and B are assumed 2D
    const size_t mblocks = A.usedblocks;
    const size_t nblocks = B.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];
    const size_t Bblock_rows = B.block_shape[0];
    const size_t Bblock_cols = B.block_shape[1];

    const size_t Astr0=A.dblock.dpstrides[0];
    const size_t Astr1=A.dblock.dpstrides[1];
    const size_t Bstr0=B.dblock.dpstrides[0];
    const size_t Bstr1=B.dblock.dpstrides[1];

    const size_t str0=C.dpstrides[0];
    const size_t str1=C.dpstrides[1];
    if(initialize_to_zero)
    {
        #pragma omp simd collapse(2)
        for(size_t i=0; i<C.dpextents[0]; i++)
            for(size_t j=0; j<C.dpextents[1]; j++)
                C.dpdata[i*str0+j*str1]=T(0);
    }


    const size_t aext0=A.dblock.dpextents[0];
    const size_t aext1=A.dblock.dpextents[1];
    const size_t bext0=B.dblock.dpextents[0];
    const size_t bext1=B.dblock.dpextents[1];
    for (size_t ia = 0; ia < mblocks; ++ia)
    {
        const size_t a_start = A.pooled_offsets_starts[ia];
        const size_t* a_off =  A.pooled_offsets_flat + a_start;

        const size_t a_row_off = a_off[0];
        const size_t a_col_off = a_off[1];

        const size_t a_rem_rows = aext0 - a_row_off;
        const size_t a_rem_cols = aext1 - a_col_off;

        const  size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
        const  size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

        for (size_t jb = 0; jb < nblocks; ++jb)
        {
            const size_t b_start = B.pooled_offsets_starts[jb];
            const size_t* b_off = B.pooled_offsets_flat + b_start;

            const size_t b_row_off = b_off[0];
            const size_t b_col_off = b_off[1];
            const size_t b_rem_rows = bext0 - b_row_off;
            const size_t b_rem_cols = bext1 - b_col_off;

            const  size_t b_tile_rows = (Bblock_rows < b_rem_rows) ? Bblock_rows : b_rem_rows;
            const  size_t b_tile_cols = (Bblock_cols < b_rem_cols) ? Bblock_cols : b_rem_cols;

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
                const  size_t global_i = a_row_off + ii;

                for (size_t jj = 0; jj < b_tile_cols; ++jj)
                {
                    const size_t global_j = b_col_off + jj;
                    T sum = T(0);
                    #pragma omp simd reduction(+:sum)
                    for (size_t kk = k_start; kk < k_end; ++kk)
                    {
                        const size_t a_index = (global_i *Astr0) +  (kk * Astr1);
                        const size_t b_index = (kk *Bstr0) +        (global_j * Bstr1);

                        sum += A.dblock.dpdata[a_index] * B.dblock.dpdata[b_index];
                    }

                    C.dpdata[global_i*str0+ global_j*str1] += sum;
                }
            }
        }
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_sparse_s(   const BlockedDataView<T>& A,  const BlockedDataView<T>& B,   DataBlock<T>& C,bool initialize_to_zero)
{
    // both A and B are assumed 2D
    const size_t mblocks = A.usedblocks;
    const size_t nblocks = B.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];
    const size_t Bblock_rows = B.block_shape[0];
    const size_t Bblock_cols = B.block_shape[1];

    const size_t Astr0=A.dblock.dpstrides[0];
    const size_t Astr1=A.dblock.dpstrides[1];
    const size_t Bstr0=B.dblock.dpstrides[0];
    const size_t Bstr1=B.dblock.dpstrides[1];

    const size_t str0=C.dpstrides[0];
    const size_t str1=C.dpstrides[1];

    const size_t aext0=A.dblock.dpextents[0];
    const size_t aext1=A.dblock.dpextents[1];
    const size_t bext0=B.dblock.dpextents[0];
    const size_t bext1=B.dblock.dpextents[1];

    if(initialize_to_zero)
        for(size_t i=0; i<C.dpextents[0]; i++)
            for(size_t j=0; j<C.dpextents[1]; j++)
                C.dpdata[i*str0+j*str1]=T(0);


    for (size_t ia = 0; ia < mblocks; ++ia)
    {
        const size_t a_start = A.pooled_offsets_starts[ia];
        const size_t* a_off =  A.pooled_offsets_flat + a_start;

        const size_t a_row_off = a_off[0];
        const size_t a_col_off = a_off[1];
        const  size_t a_rem_rows = aext0 - a_row_off;
        const  size_t a_rem_cols = aext1 - a_col_off;

        const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
        const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;


        for (size_t jb = 0; jb < nblocks; ++jb)
        {
            const size_t b_start = B.pooled_offsets_starts[jb];
            const size_t* b_off = B.pooled_offsets_flat + b_start;

            const size_t b_row_off = b_off[0];
            const size_t b_col_off = b_off[1];

            const size_t b_rem_rows = bext0 - b_row_off;
            const size_t b_rem_cols = bext1 - b_col_off;

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
                    const size_t global_j = b_col_off + jj;
                    T sum = T(0);

                    for (size_t kk = k_start; kk < k_end; ++kk)
                    {
                        const size_t a_index = (global_i * Astr0) +  (kk * Astr1);
                        const size_t b_index = (kk *Bstr0) +        (global_j * Bstr1);

                        sum += A.dblock.dpdata[a_index] * B.dblock.dpdata[b_index];
                    }

                    C.dpdata[global_i*str0+global_j*str1] += sum;
                }
            }
        }
    }
}


#pragma omp end declare target


template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_kahan_w(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    #pragma omp parallel for collapse(2) shared(A,B,C)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = T(0);
            T c=T(0);
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

template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_kahan_s(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];


    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = T(0);
            T c=T(0);
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




#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::cholesky_decomposition_w(const DataBlock<T>& A, DataBlock<T>& L, bool initialize_to_zero)
{

    const size_t n = A.dpextents[0];

    if(initialize_to_zero)
    {
        #pragma omp parallel for simd collapse(2) shared(L)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=T(0);
            }
    }

    for (size_t c = 0; c < n; ++c)
    {
        T tmp=T(0);

        #pragma omp parallel for simd reduction(-:tmp) shared(L)
        for (size_t k = 0; k < c; ++k)
        {
            const T tmp3=L(c,k);
            tmp+= tmp3 * tmp3;
        }
        tmp=A(c, c)-tmp;
        const T tmp4=sqrt(tmp);
        L(c, c) =tmp4;

        #pragma omp parallel for shared(A,L)
        for (size_t i = c + 1; i < n; ++i)
        {
            T tmp2 = A(i, c);
            #pragma omp simd reduction(-:tmp2)
            for (size_t k = 0; k < c; ++k)
            {
                tmp2 -= L(i, k) * L(c, k);
            }
            L(i, c)=tmp2/tmp4;
        }
    }

}
#pragma omp end declare target





#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::lu_decomposition_w(const  DataBlock<T>& A, DataBlock<T>& L, DataBlock<T>& U,bool initialize_to_zero)
{

    const size_t n = A.dpextents[0];


    if(initialize_to_zero)
    {
        #pragma omp parallel for simd collapse(2) shared(L,U)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=T(0);
                U(i,j)=T(0);
            }
    }

    for (size_t c = 0; c < n; ++c)
    {
        #pragma omp parallel for shared(A,L,U)
        for (size_t i = c; i < n; ++i)
        {
            T temp=A(c,i);
            #pragma omp simd reduction(-:temp)
            for (size_t k = 0; k < c; ++k)
            {
                temp -= U( k,i) * L( c,k);
            }
            U(c,i)=temp;
        }

        const T temp4=U(c,c);
        #pragma omp parallel for shared(A,U,L)
        for (size_t i = c; i < n; ++i)
        {
            T temp = A(i,c);
            #pragma omp simd reduction (-:temp)
            for (size_t k = 0; k < c; ++k)
            {
                temp -= U(k,c) * L( i,k);
            }
            L(i,c)=temp/temp4;
        }
    }

}
#pragma omp end declare target







#pragma omp begin declare target
template <typename T >
void In_Kernel_Mathfunctions<T>::qr_decomposition_w( const DataBlock<T>&A, DataBlock<T> Q, DataBlock<T> &R,bool initialize_to_zero, bool with_memmaps)
{
    const size_t n = A.dpextents[0];
    const size_t m = A.dpextents[1];


    T* __restrict tempM;

    if(with_memmaps)
        tempM=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(A.dpdatalength);
    else
        tempM=(T*)malloc(sizeof(T)*A.dpdatalength);

    size_t Mext[2]= {A.dpextents[0],A.dpextents[1]};
    size_t Mstrides[2]= {A.dpstrides[0],A.dpstrides[1]};

    DataBlock<T> M(tempM,A.dpdatalength,A.dprowmajor,A.dprank,Mext,Mstrides,false,false,false);


    if(initialize_to_zero)
    {
        #pragma omp parallel for shared(M,A,R)
        for (size_t i = 0; i < n; ++i)
        {
            #pragma omp simd
            for (size_t j = 0; j < n; ++j)
                Q(i,j) = 0;
            #pragma omp simd
            for (size_t j = 0; j < m; ++j)
            {
                M(i,j)=A(i,j);
                R(i,j) = 0;
            }
        }
    }
    else
    {
        #pragma omp parallel for shared(M,A)
        for (size_t i = 0; i < n; ++i)
        {
            #pragma omp simd
            for (size_t j = 0; j < m; ++j)
            {
                M(i,j)=A(i,j);
            }
        }
    }

    const size_t pext0=M.dpextents[0];

    for (size_t c = 0; c < m; ++c)
    {
        size_t pextv[1];
        size_t pstrv[1];
        DataBlock<T> v = M.column(c,pextv,pstrv);
        for (size_t j = 0; j < c; ++j)
        {
            size_t pextu[1];
            size_t pstru[1];

            T dot_pr=T(0);
            DataBlock<T> u = Q.column(j,pextu,pstru);
            #pragma omp parallel for simd  reduction(+:dot_pr)shared(u,v)
            for (size_t i = 0; i < pext0; ++i)
            {
                dot_pr += u(i) * v(i);
            }

            const T cdot_pr=dot_pr;
            #pragma omp  parallel for simd shared(v,u)
            for (size_t i = 0; i < pext0; ++i)
            {
                v(i) -= cdot_pr * u(i);
            }
        }
        // Normalize v
        T norm=T(0);
        #pragma omp parallel for simd reduction(+: norm) shared(v)
        for (size_t i = 0; i < pext0; ++i)
        {
            norm += v(i) * v(i);
        }

        const T normc= sqrt(norm);
        #pragma omp parallel for simd shared(v)
        for (size_t i = 0; i < pext0; ++i)
        {

            v(i)= v(i)/normc;
        }
        #pragma omp parallel for simd shared(Q,v)
        for (size_t i = 0; i < pext0; ++i)
        {
            Q(i,c) = v(i);
        }
    }

    const size_t rows = Q.dpextents[0]; // Number of rows in A and C
    const size_t cols = A.dpextents[1]; // Number of columns in B and C
    const size_t inner_dim = Q.dpextents[1]; // Number of columns in A and rows in B

    #pragma omp parallel for shared(Q,A,R)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += Q(k,i) *A(k,j);
            }
            R(i,j)= sum;
        }
    }

    if(with_memmaps)
        DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(tempM,A.dpdatalength);
    else
        free(tempM);
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::cross_product( const DataBlock<T>& vec1, const  DataBlock<T>& vec2, DataBlock<T>& res)
{
    res(0) = vec1(1) * vec2(2) - vec1(2) * vec2(1);
    res(1) = vec1(2) * vec2(0) - vec1(0) * vec2(2);
    res(2) = vec1(0) * vec2(1) - vec1(1) * vec2(0);

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w( const DataBlock<T>& A, const  DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    #pragma omp parallel for collapse(2) shared(A,B,C)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w_kahan( const DataBlock<T>& A, const  DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    #pragma omp parallel for collapse(2) shared(A,B,C)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = T(0);
            T c=T(0);
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
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_v( const DataBlock<T>& A, const  DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];


    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_s( const DataBlock<T>& A, const  DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];


    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k) *B(k,j);
            }
            C(i,j)= sum;
        }
    }
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_add_w(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    #pragma omp parallel for shared(C,A,B)
    for (size_t i = 0; i < n; ++i)
    {
        #pragma omp simd
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_add_v(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }


}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_add_s(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }


}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_subtract_w(const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    #pragma omp parallel for shared(C,A,B)
    for (size_t i = 0; i <n; ++i)
    {
        #pragma omp simd
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j) =A(i,j)-B(i,j);
        }
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_subtract_v(const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    #pragma omp simd collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j) =A(i,j)-B(i,j);
        }
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_subtract_s(const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j) =A(i,j)-B(i,j);
        }
    }

}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_w( const DataBlock<T>&M,const  DataBlock<T>& V, DataBlock<T>& C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    #pragma omp parallel for shared(M,V,C)
    for (size_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        #pragma omp simd reduction(+: sum)
        for (size_t j = 0; j <  m; ++j)
        {
            sum+= M(i, j) * V(j);
        }
        C(i)=sum;
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_v( const DataBlock<T>&M,const  DataBlock<T>& V, DataBlock<T>& C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        #pragma omp simd reduction(+: sum)
        for (size_t j = 0; j <  m; ++j)
        {
            sum+= M(i, j) * V(j);
        }
        C(i)=sum;
    }

}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_s( const DataBlock<T>&M,const  DataBlock<T>& V, DataBlock<T> &C)
{

    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        for (size_t j = 0; j <  m; ++j)
        {
            sum+= M(i, j) * V(j);
        }
        C(i)=sum;
    }

}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_kahan_s( const DataBlock<T>&M,const  DataBlock<T>& V, DataBlock<T>& C)
{

    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        T c=T(0);
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
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_kahan_w( const DataBlock<T>&M,const  DataBlock<T>& V, DataBlock<T>& C)
{

    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    #pragma omp parallel for shared(M,V)
    for (size_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        T c=T(0);
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
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_kahan_s( const DataBlock<T>&M,const T* V, DataBlock<T> &C)
{

    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        T c=T(0);
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
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_kahan_w( const DataBlock<T>&M,const T* V, DataBlock<T>& C)
{

    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    #pragma omp parallel for shared(M)
    for (size_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        T c=T(0);
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
#pragma omp end declare target





#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_s( const DataBlock<T>&M,const  T*V, DataBlock<T> & C)
{
    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        for (size_t j = 0; j <  m; ++j)
        {
            sum+= M(i, j) * V[j];
        }
        C(i)=sum;
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_v( const DataBlock<T>&M, const T*V, DataBlock<T> & C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

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
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_w( const DataBlock<T>&M, const T*V, DataBlock<T> & C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    #pragma omp parallel for shared(M,V,C)
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
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::vector_add_s( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res)
{
    const size_t n=vec1.dpextents[0];
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::vector_add_v( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res)
{
    const size_t n=vec1.dpextents[0];
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::vector_add_w( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res)
{
    const size_t n=vec1.dpextents[0];
    #pragma omp parallel for simd shared(res,vec1,vec2)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::vector_subtract_w( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res)
{
    const size_t n=vec1.dpextents[0];
    #pragma omp parallel for simd shared(res,vec1,vec2)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::vector_subtract_v( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res)
{
    const size_t n=vec1.dpextents[0];
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::vector_subtract_s( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res)
{
    const size_t n=vec1.dpextents[0];
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
T In_Kernel_Mathfunctions<T>::dot_product_s(const  DataBlock<T> &vec1, const DataBlock<T> &vec2)
{
    const size_t n=vec1.dpextents[0];
    T result=T(0);

    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
T In_Kernel_Mathfunctions<T>::dot_product_v(const  DataBlock<T> &vec1, const DataBlock<T> &vec2)
{
    const size_t n=vec1.dpextents[0];

    T result=T(0);
    #pragma omp simd reduction(+: result)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
T In_Kernel_Mathfunctions<T>::dot_product_w(const  DataBlock<T> &vec1, const DataBlock<T> &vec2)
{
    const size_t n=vec1.dpextents[0];
    T result=T(0);
    #pragma omp parallel for simd reduction(+:result)shared(vec1,vec2)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
T In_Kernel_Mathfunctions<T>::dot_product_w_kahan(const  DataBlock<T> &vec1, const DataBlock<T> &vec2)
{

    const size_t n=vec1.dpextents[0];

    int total_threads =   omp_get_max_threads();

    if(n < (size_t)total_threads)
    {
        T result = T(0);
        T c_final = T(0);
        for (int i = 0; i < n; ++i)
        {
            T y = vec1(i) * vec2(i)- c_final;
            volatile T t = result + y;
            volatile T z = t - result;
            c_final=z-y;
            result = t;
        }
        return result;
    }
    else
    {


        // Each thread gets its own local sum/compensation
        T* thread_sums = new T[total_threads];
        T* thread_cs   = new T[total_threads];

        #pragma omp parallel for simd
        for (int idx = 0; idx < total_threads; ++idx)
        {
            thread_sums[idx] = T(0);
            thread_cs[idx] = T(0);
        }

        #pragma omp parallel for
        for (int tid = 0; tid < total_threads; ++tid)
        {
            T local_sum = T(0);
            T c = T(0);

            for (size_t i = tid; i < n; i += total_threads)
            {
                T y = vec1(i) * vec2(i);
                volatile T t = local_sum + y;
                volatile T z = t - local_sum;
                c = z - y;
                local_sum = t;
            }

            thread_sums[tid] = local_sum;
            thread_cs[tid]   = c;
        }

        T result = T(0);
        T c_final = T(0);

        for (int tid = 0; tid < total_threads; ++tid)
        {
            T y = thread_sums[tid] - c_final;
            volatile T t = result + y;
            volatile T z = t - result;
            c_final=z-y;
            result = t;
        }
        delete[] thread_sums;
        delete[] thread_cs;

        return result;
    }
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_scalar_s(  const DataBlock<T>& M, const T V, DataBlock<T>& C)
{


    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V;
        }
    }

}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_scalar_v(  const DataBlock<T>& M, const T V, DataBlock<T>& C)
{


    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];

    #pragma omp simd collapse(2)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V;
        }
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_scalar_w(  const DataBlock<T>& M, const T V, DataBlock<T>& C)
{

    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];
    #pragma omp parallel for shared(C,M)
    for (size_t i = 0; i <n; ++i)
    {
        #pragma omp simd
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V;
        }
    }

}
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::vector_multiply_scalar_s( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res)
{
    const size_t n=vec.dpextents[0];
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}

#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::vector_multiply_scalar_v( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res)
{
    const size_t n=vec.dpextents[0];

    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}

#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::vector_multiply_scalar_w( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res)
{
    const size_t n=vec.dpextents[0];
    #pragma omp parallel for simd shared(res,vec)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}

#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
T  In_Kernel_Mathfunctions<T>::kahan_sum(const T *arr, size_t n)
{
    double sum = T(0);
    double c = T(0); // compensation

    for (size_t i = 0; i < n; ++i)
    {
        double y = (double)arr[i] - c;
        volatile double t = sum + y;
        volatile double z=t-sum;
        c = z - y;
        sum = t;
    }
    return (T)sum;
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
T In_Kernel_Mathfunctions<T>::neumaier_sum(const T* arr, size_t n)
{
    double sum = T(0);
    double comp = T(0);

    for (size_t i = 0; i < n; ++i)
    {
        double x = (double)arr[i];
        volatile double t = sum + x;  // volatile prevents reassociation

        if (std::fabs(sum) >= std::fabs(x))
        {
            volatile double z = (sum - t) + x; // volatile shields compensation term
            comp += z;
        }
        else
        {
            volatile double z = (x - t) + sum;
            comp += z;
        }

        sum = t;
    }
    return (T)(sum + comp);
}
#pragma omp end declare target

#endif
