#ifndef INKERNELMATHFUNCTIONShpp
#define INKERNELMATHFUNCTIONShpp

#include "datablock.h"
#include "datablocksparseutils.h"
#include "mathutilitiesdatablock.h"


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_vector_sparse(  const BlockedDataView<T>& A, const BlockedDataView<T>& x,  DataBlock<T>& y,const T CoeffB,const T Coeffy)
{
    const ptrdiff_t mblocks = A.usedblocks;
    const ptrdiff_t nblocks = x.usedblocks;

    const ptrdiff_t Ablock_rows = A.block_shape[0];
    const ptrdiff_t Ablock_cols = A.block_shape[1];
    const ptrdiff_t Xblock_size = x.block_shape[0];


    const ptrdiff_t aext0 = A.dpextents[0];
    const ptrdiff_t aext1 = A.dpextents[1];
    const ptrdiff_t xext  = x.dpextents[0];

    const ptrdiff_t ystr0 = y.dpstrides[0];


    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for(ptrdiff_t i=0; i<y.dpextents[0]; i++)
        {
            size_t index=i*ystr0;
            y.dpdata[index]=Coeffy==T(0)?T(0): Coeffy*y.dpdata[index];
        }
    }
    else  if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp  simd
        for(ptrdiff_t i=0; i<y.dpextents[0]; i++)
        {
            size_t index=i*ystr0;
            y.dpdata[index]=Coeffy==T(0)?T(0): Coeffy*y.dpdata[index];
        }
    }
    else
    {
        #pragma omp  unroll partial
        for(ptrdiff_t i=0; i<y.dpextents[0]; i++)
        {
            size_t index=i*ystr0;
            y.dpdata[index]=Coeffy==T(0)?T(0): Coeffy*y.dpdata[index];
        }
    }


    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for collapse(2)
        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {
            for (ptrdiff_t jb = 0; jb < nblocks; ++jb)
            {
                const ptrdiff_t a_start = A.pooled_offsets_starts[ia];
                const ptrdiff_t* a_off  = A.pooled_offsets_flat + a_start;

                const ptrdiff_t a_row_off = a_off[0];
                const ptrdiff_t a_col_off = a_off[1];

                const ptrdiff_t a_rem_rows = aext0 - a_row_off;
                const ptrdiff_t a_rem_cols = aext1 - a_col_off;

                const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
                const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

                const ptrdiff_t x_start = x.pooled_offsets_starts[jb];
                const ptrdiff_t* x_off  = x.pooled_offsets_flat + x_start;

                const ptrdiff_t x_off0 = x_off[0];
                const ptrdiff_t x_rem  = xext - x_off0;
                const ptrdiff_t x_tile = (Xblock_size < x_rem) ? Xblock_size : x_rem;

                const ptrdiff_t k_start = (a_col_off> x_off0) ? a_col_off:x_off0;
                const ptrdiff_t a= a_col_off + a_tile_cols;
                const ptrdiff_t b=x_off0 + x_tile;
                const ptrdiff_t k_end   =(a<b)?a:b;

                if (k_start >= k_end) continue;

                for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const ptrdiff_t global_i = a_row_off + ii;
                    T sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (ptrdiff_t kk = k_start; kk < k_end; ++kk)
                    {

                        sum += A(global_i,kk)* x(kk);
                    }
                    #pragma omp atomic update
                    y(global_i) +=CoeffB *sum;
                }
            }
        }
    }

    if constexpr (Policy == OpenMPVariant::Simd)
    {


        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {
            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];
            const ptrdiff_t* a_off  = A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];

            const ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const ptrdiff_t a_rem_cols = aext1 - a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            const ptrdiff_t a= a_col_off + a_tile_cols;
            for (ptrdiff_t jb = 0; jb < nblocks; ++jb)
            {


                const ptrdiff_t x_start = x.pooled_offsets_starts[jb];
                const ptrdiff_t* x_off  = x.pooled_offsets_flat + x_start;

                const ptrdiff_t x_off0 = x_off[0];
                const ptrdiff_t x_rem  = xext - x_off0;
                const ptrdiff_t x_tile = (Xblock_size < x_rem) ? Xblock_size : x_rem;

                const ptrdiff_t k_start = (a_col_off> x_off0) ? a_col_off:x_off0;

                const ptrdiff_t b=x_off0 + x_tile;
                const ptrdiff_t k_end   =(a<b)?a:b;

                if (k_start >= k_end) continue;
                for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const ptrdiff_t global_i = a_row_off + ii;
                    T sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (ptrdiff_t kk = k_start; kk < k_end; ++kk)
                    {

                        sum += A(global_i,kk)* x(kk);
                    }
                    y(global_i) +=CoeffB *sum;
                }
            }
        }
    }
    else
    {

        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {
            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];
            const ptrdiff_t* a_off  = A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];

            const ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const ptrdiff_t a_rem_cols = aext1 - a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            const ptrdiff_t a= a_col_off + a_tile_cols;

            for (ptrdiff_t jb = 0; jb < nblocks; ++jb)
            {


                const ptrdiff_t x_start = x.pooled_offsets_starts[jb];
                const ptrdiff_t* x_off  = x.pooled_offsets_flat + x_start;

                const ptrdiff_t x_off0 = x_off[0];
                const ptrdiff_t x_rem  = xext - x_off0;
                const ptrdiff_t x_tile = (Xblock_size < x_rem) ? Xblock_size : x_rem;

                const ptrdiff_t k_start = (a_col_off> x_off0) ? a_col_off:x_off0;

                const ptrdiff_t b=x_off0 + x_tile;
                const ptrdiff_t k_end   =(a<b)?a:b;

                if (k_start >= k_end) continue;
                for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const ptrdiff_t global_i = a_row_off + ii;
                    T sum = 0;
                    #pragma omp unroll partial
                    for (ptrdiff_t kk = k_start; kk < k_end; ++kk)
                    {
                        sum += A(global_i,kk)* x(kk);
                    }
                    y(global_i) +=CoeffB *sum;
                }
            }
        }
    }
}
#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_vector_sparse( const BlockedDataView<T>& A, const DataBlock<T>& x, DataBlock<T>& y,const T CoeffB,const T Coeffy)
{
    const ptrdiff_t mblocks = A.usedblocks;

    const ptrdiff_t Ablock_rows = A.block_shape[0];
    const ptrdiff_t Ablock_cols = A.block_shape[1];



    const ptrdiff_t aext0 = A.dpextents[0];
    const ptrdiff_t aext1 = A.dpextents[1];

    const ptrdiff_t ystr0 = y.dpstrides[0];



    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for(ptrdiff_t i=0; i<y.dpextents[0]; i++)
        {
            size_t index=i*ystr0;
            y.dpdata[index]=Coeffy==T(0)?T(0): Coeffy*y.dpdata[index];
        }

    }
    else  if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd
        for(ptrdiff_t i=0; i<y.dpextents[0]; i++)
        {
            size_t index=i*ystr0;
            y.dpdata[index]=Coeffy==T(0)?T(0): Coeffy*y.dpdata[index];
        }
    }
    else
    {
        #pragma omp  unroll partial
        for(ptrdiff_t i=0; i<y.dpextents[0]; i++)
        {
            size_t index=i*ystr0;
            y.dpdata[index]=Coeffy==T(0)?T(0): Coeffy*y.dpdata[index];
        }
    }



    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for
        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {

            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];
            const ptrdiff_t* a_off  = A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];

            const ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const ptrdiff_t a_rem_cols = aext1 - a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

            for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const ptrdiff_t global_i = a_row_off + ii;
                T sum = 0;
                #pragma omp simd reduction(+:sum)
                for (ptrdiff_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const ptrdiff_t global_k = a_col_off + kk;

                    sum += A(global_i,global_k) * x(global_k);
                }
                #pragma omp atomic update
                y(global_i) +=CoeffB* sum;
            }
        }
    }

    if constexpr (Policy == OpenMPVariant::Simd)
    {

        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {

            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];
            const ptrdiff_t* a_off  = A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];

            const ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const ptrdiff_t a_rem_cols = aext1 - a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const ptrdiff_t global_i = a_row_off + ii;
                T sum = 0;
                #pragma omp simd reduction(+:sum)
                for (ptrdiff_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const ptrdiff_t global_k = a_col_off + kk;

                    sum += A(global_i,global_k) * x(global_k);
                }
                y(global_i) +=CoeffB* sum;
            }
        }
    }
    else
    {

        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {

            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];
            const ptrdiff_t* a_off  = A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];

            const ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const ptrdiff_t a_rem_cols = aext1 - a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

            for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const ptrdiff_t global_i = a_row_off + ii;
                T sum = 0;
                #pragma omp unroll partial
                for (ptrdiff_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const ptrdiff_t global_k = a_col_off + kk;

                    sum += A(global_i,global_k) * x(global_k);
                }
                y(global_i) +=CoeffB* sum;
            }
        }
    }
}

#pragma omp end declare target

template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_dot_sparse( const BlockedDataView<T>& A,  const DataBlock<T>& B, DataBlock<T>& C,const T CoeffB,const T CoeffC)
{
    const ptrdiff_t mblocks = A.usedblocks;

    const ptrdiff_t Ablock_rows = A.block_shape[0];
    const ptrdiff_t Ablock_cols = A.block_shape[1];


    const ptrdiff_t Cstr0 = C.dpstrides[0];
    const ptrdiff_t Cstr1 = C.dpstrides[1];

    const ptrdiff_t aext0 = A.dpextents[0];
    const ptrdiff_t aext1 = A.dpextents[1];
    const ptrdiff_t bext0 = B.dpextents[0]; // must equal aext1
    const ptrdiff_t bext1 = B.dpextents[1];


    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for(ptrdiff_t i=0; i<C.dpextents[0]; i++)
        {
            for(ptrdiff_t j=0; j<C.dpextents[1]; j++)
            {
                const size_t index=i*Cstr0+j*Cstr1;
                C.dpdata[index]=CoeffC==T(0)?T(0): CoeffC*C.dpdata[index];
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for(ptrdiff_t i=0; i<C.dpextents[0]; i++)
        {
            for(ptrdiff_t j=0; j<C.dpextents[1]; j++)
            {
                const size_t index=i*Cstr0+j*Cstr1;
                C.dpdata[index]=CoeffC==T(0)?T(0): CoeffC*C.dpdata[index];
            }
        }
    }
    else
    {
        for(ptrdiff_t i=0; i<C.dpextents[0]; i++)
        {
            #pragma omp unroll partial
            for(ptrdiff_t j=0; j<C.dpextents[1]; j++)
            {
                const size_t index=i*Cstr0+j*Cstr1;
                C.dpdata[index]=CoeffC==T(0)?T(0): CoeffC*C.dpdata[index];
            }
        }
    }

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for
        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {
            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];
            const ptrdiff_t* a_off  = A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];

            const ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const ptrdiff_t a_rem_cols = aext1 - a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const ptrdiff_t global_i = a_row_off + ii;
                for (ptrdiff_t jj = 0; jj < bext1; ++jj)
                {
                    T sum = T(0);
                    #pragma omp simd reduction(+:sum)
                    for (ptrdiff_t kk = 0; kk < a_tile_cols; ++kk)
                    {
                        const ptrdiff_t global_k = a_col_off + kk;
                        sum += A(global_i,global_k) * B(global_k,jj);
                    }
                    #pragma omp atomic update
                    C(global_i,jj) +=CoeffB* sum;
                }
            }
        }
    }

    else  if constexpr (Policy == OpenMPVariant::Simd)
    {
        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {

            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];
            const ptrdiff_t* a_off  = A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];

            const ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const ptrdiff_t a_rem_cols = aext1 - a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const ptrdiff_t global_i = a_row_off + ii;
                for (ptrdiff_t jj = 0; jj < bext1; ++jj)
                {
                    T sum = T(0);
                    #pragma omp simd reduction(+:sum)
                    for (ptrdiff_t kk = 0; kk < a_tile_cols; ++kk)
                    {
                        const ptrdiff_t global_k = a_col_off + kk;
                        sum += A(global_i,global_k) * B(global_k,jj);
                    }
                    C(global_i,jj) +=CoeffB* sum;
                }
            }
        }
    }
    else
    {
        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {

            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];
            const ptrdiff_t* a_off  = A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];

            const ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const ptrdiff_t a_rem_cols = aext1 - a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
            {
                const ptrdiff_t global_i = a_row_off + ii;
                for (ptrdiff_t jj = 0; jj < bext1; ++jj)
                {
                    T sum = T(0);
                    #pragma omp unroll partial
                    for (ptrdiff_t kk = 0; kk < a_tile_cols; ++kk)
                    {
                        const ptrdiff_t global_k = a_col_off + kk;
                        sum += A(global_i,global_k) * B(global_k,jj);
                    }
                    C(global_i,jj) += CoeffB*sum;
                }
            }
        }

    }

}


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_dot_sparse( const BlockedDataView<T>& A, const BlockedDataView<T>& B, DataBlock<T>& C, const T CoeffB,const T CoeffC)
{
    const ptrdiff_t mblocks = A.usedblocks;
    const ptrdiff_t nblocks = B.usedblocks;

    const ptrdiff_t Ablock_rows = A.block_shape[0];
    const ptrdiff_t Ablock_cols = A.block_shape[1];
    const ptrdiff_t Bblock_rows = B.block_shape[0];
    const ptrdiff_t Bblock_cols = B.block_shape[1];

    const ptrdiff_t Cstr0=C.dpstrides[0];
    const ptrdiff_t Cstr1=C.dpstrides[1];
    const ptrdiff_t aext0=A.dpextents[0];
    const ptrdiff_t aext1=A.dpextents[1];
    const ptrdiff_t bext0=B.dpextents[0];
    const ptrdiff_t bext1=B.dpextents[1];


    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for(ptrdiff_t i=0; i<C.dpextents[0]; i++)
        {
            for(ptrdiff_t j=0; j<C.dpextents[1]; j++)
            {
                const size_t index=i*Cstr0+j*Cstr1;
                C.dpdata[index]=CoeffC==T(0)?T(0): CoeffC*C.dpdata[index];
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for(ptrdiff_t i=0; i<C.dpextents[0]; i++)
        {
            for(ptrdiff_t j=0; j<C.dpextents[1]; j++)
            {
                const size_t index=i*Cstr0+j*Cstr1;
                C.dpdata[index]=CoeffC=T(0)?T(0): CoeffC*C.dpdata[index];
            }
        }
    }
    else
    {
        for(ptrdiff_t i=0; i<C.dpextents[0]; i++)
        {
            #pragma omp unroll partial
            for(ptrdiff_t j=0; j<C.dpextents[1]; j++)
            {
                const size_t index=i*Cstr0+j*Cstr1;
                C.dpdata[index]=CoeffC==T(0)?T(0): CoeffC*C.dpdata[index];
            }

        }
    }


    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for collapse(2)
        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {
            for (ptrdiff_t jb = 0; jb < nblocks; ++jb)
            {
                const ptrdiff_t a_start = A.pooled_offsets_starts[ia];

                const ptrdiff_t* a_off =  A.pooled_offsets_flat + a_start;

                const ptrdiff_t a_row_off = a_off[0];
                const ptrdiff_t a_col_off = a_off[1];

                const ptrdiff_t a_rem_rows = aext0 - a_row_off;
                const ptrdiff_t a_rem_cols = aext1- a_col_off;

                const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
                const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

                const ptrdiff_t b_start = B.pooled_offsets_starts[jb];

                const ptrdiff_t* b_off = B.pooled_offsets_flat + b_start;
                const ptrdiff_t b_row_off = b_off[0];
                const ptrdiff_t b_col_off = b_off[1];

                const ptrdiff_t b_rem_rows =bext0 - b_row_off;
                const ptrdiff_t b_rem_cols =bext1 - b_col_off;

                const ptrdiff_t b_tile_rows = (Bblock_rows < b_rem_rows) ? Bblock_rows : b_rem_rows;
                const ptrdiff_t b_tile_cols = (Bblock_cols < b_rem_cols) ? Bblock_cols : b_rem_cols;

                const ptrdiff_t a_k_start = a_col_off;
                const ptrdiff_t a_k_end   = a_col_off + a_tile_cols;

                const ptrdiff_t b_k_start = b_row_off;
                const ptrdiff_t b_k_end   = b_row_off + b_tile_rows;

                const ptrdiff_t k_start = (a_k_start >   b_k_start)  ?   a_k_start:   b_k_start;
                const ptrdiff_t k_end   = (a_k_end   <   b_k_end)    ?   a_k_end:     b_k_end;

                if (k_start >= k_end)
                {
                    continue;
                }

                for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const ptrdiff_t global_i = a_row_off + ii;
                    for (ptrdiff_t jj = 0; jj < b_tile_cols; ++jj)
                    {
                        const  ptrdiff_t global_j = b_col_off + jj;
                        T sum = T(0);
                        #pragma omp simd reduction(+:sum)
                        for (ptrdiff_t kk = k_start; kk < k_end; ++kk)
                        {
                            sum += A(global_i,kk)* B(kk,global_j);
                        }
                        #pragma omp atomic update
                        C(global_i, global_j) += CoeffB*sum;
                    }
                }
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {


        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {
            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];

            const ptrdiff_t* a_off =  A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];

            const ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const ptrdiff_t a_rem_cols = aext1- a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            const ptrdiff_t a_k_start = a_col_off;
            const ptrdiff_t a_k_end   = a_col_off + a_tile_cols;

            for (ptrdiff_t jb = 0; jb < nblocks; ++jb)
            {

                const ptrdiff_t b_start = B.pooled_offsets_starts[jb];

                const ptrdiff_t* b_off = B.pooled_offsets_flat + b_start;
                const ptrdiff_t b_row_off = b_off[0];
                const ptrdiff_t b_col_off = b_off[1];

                const ptrdiff_t b_rem_rows =bext0 - b_row_off;
                const ptrdiff_t b_rem_cols =bext1 - b_col_off;

                const ptrdiff_t b_tile_rows = (Bblock_rows < b_rem_rows) ? Bblock_rows : b_rem_rows;
                const ptrdiff_t b_tile_cols = (Bblock_cols < b_rem_cols) ? Bblock_cols : b_rem_cols;


                const ptrdiff_t b_k_start = b_row_off;
                const ptrdiff_t b_k_end   = b_row_off + b_tile_rows;

                const ptrdiff_t k_start = (a_k_start >   b_k_start)  ?   a_k_start:   b_k_start;
                const ptrdiff_t k_end   = (a_k_end   <   b_k_end)    ?   a_k_end:     b_k_end;

                if (k_start >= k_end)
                {
                    continue;
                }
                for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const ptrdiff_t global_i = a_row_off + ii;
                    for (ptrdiff_t jj = 0; jj < b_tile_cols; ++jj)
                    {
                        const  ptrdiff_t global_j = b_col_off + jj;
                        T sum = T(0);
                        #pragma omp simd reduction(+:sum)
                        for (ptrdiff_t kk = k_start; kk < k_end; ++kk)
                        {
                            sum += A(global_i,kk)* B(kk,global_j);
                        }
                        C(global_i, global_j) +=CoeffB* sum;
                    }
                }
            }
        }
    }
    else
    {
        for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
        {
            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];

            const ptrdiff_t* a_off =  A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];

            const ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const ptrdiff_t a_rem_cols = aext1- a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;
            const ptrdiff_t a_k_start = a_col_off;
            const ptrdiff_t a_k_end   = a_col_off + a_tile_cols;
            for (ptrdiff_t jb = 0; jb < nblocks; ++jb)
            {

                const ptrdiff_t b_start = B.pooled_offsets_starts[jb];

                const ptrdiff_t* b_off = B.pooled_offsets_flat + b_start;
                const ptrdiff_t b_row_off = b_off[0];
                const ptrdiff_t b_col_off = b_off[1];

                const ptrdiff_t b_rem_rows =bext0 - b_row_off;
                const ptrdiff_t b_rem_cols =bext1 - b_col_off;

                const ptrdiff_t b_tile_rows = (Bblock_rows < b_rem_rows) ? Bblock_rows : b_rem_rows;
                const ptrdiff_t b_tile_cols = (Bblock_cols < b_rem_cols) ? Bblock_cols : b_rem_cols;


                const ptrdiff_t b_k_start = b_row_off;
                const ptrdiff_t b_k_end   = b_row_off + b_tile_rows;

                const ptrdiff_t k_start = (a_k_start >   b_k_start)  ?   a_k_start:   b_k_start;
                const ptrdiff_t k_end   = (a_k_end   <   b_k_end)    ?   a_k_end:     b_k_end;

                if (k_start >= k_end)
                {
                    continue;
                }
                for (ptrdiff_t ii = 0; ii < a_tile_rows; ++ii)
                {
                    const ptrdiff_t global_i = a_row_off + ii;
                    for (ptrdiff_t jj = 0; jj < b_tile_cols; ++jj)
                    {
                        const  ptrdiff_t global_j = b_col_off + jj;
                        T sum = T(0);
                        #pragma omp unroll partial
                        for (ptrdiff_t kk = k_start; kk < k_end; ++kk)
                        {
                            sum += A(global_i,kk)* B(kk,global_j);
                        }

                        C(global_i, global_j) +=CoeffB* sum;
                    }
                }
            }
        }
    }

}
#pragma omp end declare target





#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::cholesky_decomposition(const DataBlock<T>& A, DataBlock<T>& L,bool initialize_to_zero)
{
    const ptrdiff_t n = A.dpextents[0];
    L.dpconjugate=false;
    if(initialize_to_zero)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                }
            }
        }
        else if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                }
            }
        }
        else
        {
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                #pragma omp unroll partial
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                }
            }
        }
    }

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        for (ptrdiff_t c = 0; c < n; ++c)
        {
            T tmp=T(0);

            #pragma omp  parallel for simd reduction(+:tmp)
            for (ptrdiff_t k = 0; k < c; ++k)
            {
                const T tmp3=L(c,k);
                tmp+= tmp3 *cond_conj( tmp3);
            }


            tmp=A(c, c)-tmp;
            const T tmp4=sqrt(tmp);
            L(c, c) =tmp4;

            #pragma omp parallel for
            for (ptrdiff_t i = c + 1; i < n; ++i)
            {
                T tmp2 =0;
                #pragma omp simd reduction(+:tmp2)
                for (ptrdiff_t k = 0; k < c; ++k)
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
        for (ptrdiff_t c = 0; c < n; ++c)
        {
            T tmp=T(0);

            #pragma omp simd reduction(+:tmp)
            for (ptrdiff_t k = 0; k < c; ++k)
            {
                const T tmp3=L(c,k);
                tmp+= tmp3 *cond_conj( tmp3);
            }


            tmp=A(c, c)-tmp;
            const T tmp4=sqrt(tmp);
            L(c, c) =tmp4;

            for (ptrdiff_t i = c + 1; i < n; ++i)
            {
                T tmp2 =0;
                #pragma omp simd reduction(+:tmp2)
                for (ptrdiff_t k = 0; k < c; ++k)
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
        for (ptrdiff_t c = 0; c < n; ++c)
        {
            T tmp=T(0);

            #pragma omp unroll partial
            for (ptrdiff_t k = 0; k < c; ++k)
            {
                const T tmp3=L(c,k);
                tmp+= tmp3 *cond_conj( tmp3);
            }


            tmp=A(c, c)-tmp;
            const T tmp4=sqrt(tmp);
            L(c, c) =tmp4;

            for (ptrdiff_t i = c + 1; i < n; ++i)
            {
                T tmp2 =0;
                #pragma omp unroll partial
                for (ptrdiff_t k = 0; k < c; ++k)
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

    const ptrdiff_t n = A.dpextents[0];
    L.dpconjugate=false;
    U.dpconjugate=false;

    if(initialize_to_zero)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp  parallel for simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                    U(i,j)=T(0);
                }
            }
        }
        else if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp  simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                    U(i,j)=T(0);
                }
            }
        }
        else
        {
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                #pragma omp unroll partial
                for (ptrdiff_t j = 0; j <n; ++j)
                {
                    L(i,j)=T(0);
                    U(i,j)=T(0);
                }
            }

        }
    }
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        for (ptrdiff_t c = 0; c < n; ++c)
        {
            #pragma omp parallel for
            for (ptrdiff_t i = c; i < n; ++i)
            {
                T temp=T(0);
                #pragma omp  simd reduction(+:temp)
                for (ptrdiff_t k = 0; k < c; ++k)
                {
                    temp += U( k,i) * L( c,k);
                }
                temp=A(c,i)-temp;
                U(c,i)=temp;
            }

            const T temp4=U(c,c);
            #pragma omp parallel for
            for (ptrdiff_t i = c; i < n; ++i)
            {
                T temp =T(0);
                #pragma omp simd reduction(+:temp)
                for (ptrdiff_t k = 0; k < c; ++k)
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
        for (ptrdiff_t c = 0; c < n; ++c)
        {
            for (ptrdiff_t i = c; i < n; ++i)
            {
                T temp=T(0);
                #pragma omp  simd reduction(+:temp)
                for (ptrdiff_t k = 0; k < c; ++k)
                {
                    temp += U( k,i) * L( c,k);
                }
                temp=A(c,i)-temp;
                U(c,i)=temp;
            }

            const T temp4=U(c,c);
            for (ptrdiff_t i = c; i < n; ++i)
            {
                T temp =T(0);
                #pragma omp simd reduction(+:temp)
                for (ptrdiff_t k = 0; k < c; ++k)
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
        for (ptrdiff_t c = 0; c < n; ++c)
        {
            for (ptrdiff_t i = c; i < n; ++i)
            {
                T temp=T(0);
                #pragma omp unroll partial
                for (ptrdiff_t k = 0; k < c; ++k)
                {
                    temp += U( k,i) * L( c,k);
                }
                temp=A(c,i)-temp;
                U(c,i)=temp;
            }

            const T temp4=U(c,c);
            for (ptrdiff_t i = c; i < n; ++i)
            {
                T temp =T(0);
                #pragma omp unroll partial
                for (ptrdiff_t k = 0; k < c; ++k)
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
void In_Kernel_Mathfunctions::qr_decomposition( const DataBlock<T>&A, DataBlock<T> &Q, DataBlock<T> &R,bool initialize_to_zero, bool with_memmaps)
{
    const ptrdiff_t n = A.dpextents[0];
    const ptrdiff_t m = A.dpextents[1];

    Q.dpconjugate=false;
    R.dpconjugate=false;
    T* tempM;

    if(with_memmaps)
        tempM=Host_Memory_Functions::create_temp_mmap<T>(A.dpdatalength);
    else
        tempM=(T*)omp_alloc(sizeof(T)*A.dpdatalength,omp_default_mem_alloc);


    ptrdiff_t Mext[2]= {A.dpextents[0],A.dpextents[1]};
    ptrdiff_t Mstrides[2]= {A.dpstrides[0],A.dpstrides[1]};
    DataBlockConfig mconf({.dprowmajor=A.dpconfig.dprowmajor,
                           .pmemmap=with_memmaps,
                           .data_is_devptr=false,
                           .devicenum=-INT_MAX
                          });
    DataBlock<T> M(tempM,A.dpdatalength,A.dprank,Mext,Mstrides,mconf);


    if(initialize_to_zero)
    {
        if constexpr (Policy == OpenMPVariant::ParallelSimd)
        {
            #pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (ptrdiff_t j = 0; j < n; ++j)
                    Q(i,j) = 0;
                #pragma omp  simd
                for (ptrdiff_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                    R(i,j) = 0;
                }
            }
        }
        else  if constexpr (Policy == OpenMPVariant::Simd)
        {
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (ptrdiff_t j = 0; j < n; ++j)
                    Q(i,j) = 0;
                #pragma omp  simd
                for (ptrdiff_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                    R(i,j) = 0;
                }
            }
        }
        else
        {
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                #pragma omp unroll partial
                for (ptrdiff_t j = 0; j < n; ++j)
                    Q(i,j) = 0;
                #pragma omp unroll partial
                for (ptrdiff_t j = 0; j < m; ++j)
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
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                }
            }
        }
        else if constexpr (Policy == OpenMPVariant::Simd)
        {
            #pragma omp simd collapse(2)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                }
            }
        }
        else
        {
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                #pragma omp unroll partial
                for (ptrdiff_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                }
            }
        }
    }

    const ptrdiff_t pext0=M.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        for (ptrdiff_t c = 0; c < m; ++c)
        {
            ptrdiff_t pextv[1];
            ptrdiff_t pstrv[1];
            DataBlock<T> v = DataBlockUtilities::matrix_column(M,c,pextv,pstrv);
            for (ptrdiff_t j = 0; j < c; ++j)
            {
                ptrdiff_t pextu[1];
                ptrdiff_t pstru[1];

                T dot_pr=T(0);
                DataBlock<T> u = DataBlockUtilities::matrix_column(Q,j,pextu,pstru);
                #pragma omp parallel for simd reduction(+:dot_pr)
                for (ptrdiff_t i = 0; i < pext0; ++i)
                {
                    dot_pr += cond_conj(u(i)) * v(i);
                }

                const T cdot_pr=dot_pr;
                #pragma omp parallel for simd
                for (ptrdiff_t i = 0; i < pext0; ++i)
                {
                    v(i) -= cdot_pr * u(i);
                }
            }
            T norm=T(0);
            #pragma omp parallel for simd reduction(+:norm)
            for (ptrdiff_t i = 0; i < pext0; ++i)
            {
                norm += cond_conj(v(i)) * v(i);
            }

            const T normc= sqrt(norm);
            #pragma omp parallel for simd
            for (ptrdiff_t i = 0; i < pext0; ++i)
            {
                Q(i,c) = v(i)/normc;
            }
        }

        const ptrdiff_t rows = Q.dpextents[0];
        const ptrdiff_t cols = A.dpextents[1];
        const ptrdiff_t inner_dim = Q.dpextents[1];

        #pragma omp parallel for collapse(2)
        for (ptrdiff_t i = 0; i < rows; ++i)
        {
            for (ptrdiff_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp simd reduction(+:sum)
                for (ptrdiff_t k = 0; k < inner_dim; ++k)
                {
                    sum += cond_conj(Q(k,i)) *A(k,j);
                }
                R(i,j)= sum;
            }
        }
    }
    else  if constexpr (Policy == OpenMPVariant::Simd)
    {
        for (ptrdiff_t c = 0; c < m; ++c)
        {
            ptrdiff_t pextv[1],pstrv[1];
            DataBlock<T> v = M.matrix_column(c,pextv,pstrv);
            for (ptrdiff_t j = 0; j < c; ++j)
            {
                ptrdiff_t pextu[1],pstru[1];

                T dot_pr=T(0);
                DataBlock<T> u = Q.matrix_column(j,pextu,pstru);
                #pragma omp simd reduction(+:dot_pr)
                for (ptrdiff_t i = 0; i < pext0; ++i)
                {
                    dot_pr += cond_conj(u(i)) * v(i);
                }

                const T cdot_pr=dot_pr;
                #pragma omp simd
                for (ptrdiff_t i = 0; i < pext0; ++i)
                {
                    v(i) -= cdot_pr * u(i);
                }
            }
            // Normalize v
            T norm=T(0);
            #pragma omp simd reduction(+:norm)
            for (ptrdiff_t i = 0; i < pext0; ++i)
            {
                norm += cond_conj(v(i)) * v(i);
            }

            const T normc= sqrt(norm);
            #pragma omp simd
            for (ptrdiff_t i = 0; i < pext0; ++i)
            {
                Q(i,c) = v(i)/normc;
            }
        }

        const ptrdiff_t rows = Q.dpextents[0];
        const ptrdiff_t cols = A.dpextents[1];
        const ptrdiff_t inner_dim = Q.dpextents[1];
        #pragma omp tile sizes(16, 16)
        for (ptrdiff_t i = 0; i < rows; ++i)
        {
            for (ptrdiff_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp  simd reduction(+:sum)
                for (ptrdiff_t k = 0; k < inner_dim; ++k)
                {
                    sum += cond_conj(Q(k,i)) *A(k,j);
                }
                R(i,j)= sum;
            }
        }
    }
    else
    {
        for (ptrdiff_t c = 0; c < m; ++c)
        {
            ptrdiff_t pextv[1],pstrv[1];
            DataBlock<T> v = DataBlockUtilities::matrix_column(M,c,pextv,pstrv);
            for (ptrdiff_t j = 0; j < c; ++j)
            {
                ptrdiff_t pextu[1],pstru[1];

                T dot_pr=T(0);
                DataBlock<T> u = DataBlockUtilities::matrix_column(Q,j,pextu,pstru);
                #pragma omp unroll partial
                for (ptrdiff_t i = 0; i < pext0; ++i)
                {
                    dot_pr += cond_conj(u(i)) * v(i);
                }

                const T cdot_pr=dot_pr;
                #pragma omp unroll partial
                for (ptrdiff_t i = 0; i < pext0; ++i)
                {
                    v(i) -= cdot_pr * u(i);
                }
            }
            T norm=T(0);
            #pragma omp unroll partial
            for (ptrdiff_t i = 0; i < pext0; ++i)
            {
                norm += cond_conj(v(i)) * v(i);
            }

            const T normc= sqrt(norm);
            #pragma omp unroll partial
            for (ptrdiff_t i = 0; i < pext0; ++i)
            {
                Q(i,c) = v(i)/normc;
            }
        }

        const ptrdiff_t rows = Q.dpextents[0];
        const ptrdiff_t cols = A.dpextents[1];
        const ptrdiff_t inner_dim = Q.dpextents[1];

        #pragma omp tile sizes(16, 16)
        for (ptrdiff_t i = 0; i < rows; ++i)
        {
            for (ptrdiff_t j = 0; j < cols; ++j)
            {
                T sum = 0;
                #pragma omp unroll partial
                for (ptrdiff_t k = 0; k < inner_dim; ++k)
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
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_dot( const DataBlock<T>& A, const  DataBlock<T>& B,DataBlock<T>& C, const T CoeffB,const T CoeffC )
{
    const ptrdiff_t rows=A.dpextents[0];
    const ptrdiff_t cols=B.dpextents[1];
    const ptrdiff_t inner_dim=A.dpextents[1];

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for collapse(2)
        for (ptrdiff_t i = 0; i < rows; ++i)
        {
            for (ptrdiff_t j = 0; j < cols; ++j)
            {
                T sum =T(0);
                #pragma omp simd reduction(+:sum)
                for (ptrdiff_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k) *B(k,j);
                }
                C(i,j) =CoeffC == T(0)? CoeffB * sum: CoeffC * C(i,j) + CoeffB * sum;
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp tile sizes(16,16)
        for (ptrdiff_t i = 0; i < rows; ++i)
        {
            for (ptrdiff_t j = 0; j < cols; ++j)
            {
                T sum =T(0);
                #pragma omp simd reduction(+:sum)
                for (ptrdiff_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k) *B(k,j);
                }
                C(i,j) =CoeffC == T(0)? CoeffB * sum: CoeffC * C(i,j) + CoeffB * sum;
            }
        }
    }
    else
    {
        #pragma omp tile sizes(16,16)
        for (ptrdiff_t i = 0; i < rows; ++i)
        {
            for (ptrdiff_t j = 0; j < cols; ++j)
            {
                T sum =T(0);
                #pragma omp unroll partial
                for (ptrdiff_t k = 0; k < inner_dim; ++k)
                {
                    sum += A(i,k) *B(k,j);
                }

                C(i,j) =CoeffC == T(0)? CoeffB * sum: CoeffC * C(i,j) + CoeffB * sum;
            }
        }
    }
}

#pragma omp end declare target

#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_dot_kahan(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,const T CoeffB,const T CoeffC)
{
    const ptrdiff_t rows=A.dpextents[0];
    const ptrdiff_t cols=B.dpextents[1];
    const ptrdiff_t inner_dim=A.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for collapse(2)
        for (ptrdiff_t i = 0; i < rows; ++i)
        {
            for (ptrdiff_t j = 0; j < cols; ++j)
            {
                T sum = T(0);
                T c=T(0);
                #pragma omp unroll partial
                for (ptrdiff_t k = 0; k < inner_dim; ++k)
                {
                    T y =  A(i,k) *B(k,j) - c;
                    volatile T t = sum + y;
                    volatile T z = t - sum;
                    c = z - y;
                    sum = t;
                }
                C(i,j) =CoeffC == T(0)? CoeffB * sum: CoeffC * C(i,j) + CoeffB * sum;
            }
        }
    }
    else
    {
        #pragma omp tile sizes(16,16)
        for (ptrdiff_t i = 0; i < rows; ++i)
        {
            for (ptrdiff_t j = 0; j < cols; ++j)
            {
                T sum = T(0);
                T c=T(0);
                #pragma omp unroll partial
                for (ptrdiff_t k = 0; k < inner_dim; ++k)
                {
                    T y =  A(i,k) *B(k,j) - c;
                    volatile T t = sum + y;
                    volatile T z = t - sum;
                    c = z - y;
                    sum = t;
                }
                C(i,j) =CoeffC == T(0)? CoeffB * sum: CoeffC * C(i,j) + CoeffB * sum;
            }
        }
    }
}
#pragma omp end declare target

#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_linear_combination(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,const T CoeffA,const T CoeffB, const T CoeffC)
{
    const ptrdiff_t n=A.dpextents[0];
    const ptrdiff_t m=A.dpextents[1];

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            for (ptrdiff_t j = 0; j <m ; ++j)
            {
                C(i,j) =CoeffC == T(0)? CoeffA*A(i,j)+CoeffB*B(i,j): CoeffC*C(i,j)+CoeffA*A(i,j)+CoeffB*B(i,j);
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            for (ptrdiff_t j = 0; j <m ; ++j)
            {
                C(i,j) =CoeffC == T(0)? CoeffA*A(i,j)+CoeffB*B(i,j): CoeffC*C(i,j)+CoeffA*A(i,j)+CoeffB*B(i,j);
            }
        }
    }
    else
    {
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            #pragma omp unroll partial
            for (ptrdiff_t j = 0; j <m ; ++j)
            {
                C(i,j) =CoeffC == T(0)? CoeffA*A(i,j)+CoeffB*B(i,j): CoeffC*C(i,j)+CoeffA*A(i,j)+CoeffB*B(i,j);
            }
        }
    }


}
#pragma omp end declare target





#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_linear_combination(const DataBlock<T>& A, DataBlock<T>& C,const T CoeffA, const T CoeffC)
{
    const ptrdiff_t n=A.dpextents[0];
    const ptrdiff_t m=A.dpextents[1];

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            for (ptrdiff_t j = 0; j <m ; ++j)
            {
                C(i,j) =CoeffC == T(0)? CoeffA*A(i,j): CoeffC*C(i,j)+CoeffA*A(i,j);
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            for (ptrdiff_t j = 0; j <m ; ++j)
            {
                C(i,j) =CoeffC == T(0)? CoeffA*A(i,j): CoeffC*C(i,j)+CoeffA*A(i,j);
            }
        }
    }
    else
    {
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            #pragma omp unroll partial
            for (ptrdiff_t j = 0; j <m ; ++j)
            {
                C(i,j) =CoeffC == T(0)? CoeffA*A(i,j): CoeffC*C(i,j)+CoeffA*A(i,j);
            }
        }
    }


}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_vector( const DataBlock<T>&M,const  DataBlock<T>& V, DataBlock<T>& C,const T CoeffV,const T CoeffC)
{


    const ptrdiff_t n= M.dpextents[0];
    const ptrdiff_t m=M.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for
        for (ptrdiff_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            #pragma omp simd reduction(+:sum)
            for (ptrdiff_t j = 0; j <  m; ++j)
            {
                sum+= M(i, j) * V(j);
            }
            C(i)=CoeffC == T(0)?CoeffV*sum: CoeffC*C(i)+CoeffV*sum;
        }

    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {

        for (ptrdiff_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            #pragma omp simd reduction(+:sum)
            for (ptrdiff_t j = 0; j <  m; ++j)
            {
                sum+= M(i, j) * V(j);
            }
            C(i)=CoeffC == T(0)?CoeffV*sum: CoeffC*C(i)+CoeffV*sum;
        }

    }
    else
    {

        for (ptrdiff_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            #pragma omp unroll partial
            for (ptrdiff_t j = 0; j <  m; ++j)
            {
                sum+= M(i, j) * V(j);
            }
            C(i)=CoeffC == T(0)?CoeffV*sum: CoeffC*C(i)+CoeffV*sum;
        }
    }


}
#pragma omp end declare target



#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_vector_kahan( const DataBlock<T>&M,const  DataBlock<T>& V, DataBlock<T>& C,const T CoeffV,const T CoeffC)
{

    const ptrdiff_t n= M.dpextents[0];
    const ptrdiff_t m=M.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for
        for (ptrdiff_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            T c=T(0);
            #pragma omp unroll partial
            for (ptrdiff_t j = 0; j <  m; ++j)
            {
                T y = M(i, j) * V(j) - c;
                volatile T t = sum + y;
                volatile T z = t - sum;
                c = z - y;
                sum = t;
            }
            C(i)=CoeffC == T(0)?CoeffV*sum: CoeffC*C(i)+CoeffV*sum;
        }
    }
    else
    {

        for (ptrdiff_t i = 0; i <n; ++i)
        {
            T sum=T(0);
            T c=T(0);
            #pragma omp unroll partial
            for (ptrdiff_t j = 0; j <  m; ++j)
            {
                T y = M(i, j) * V(j) - c;
                volatile T t = sum + y;
                volatile T z = t - sum;
                c = z - y;
                sum = t;
            }
            C(i)=CoeffC == T(0)?CoeffV*sum: CoeffC*C(i)+CoeffV*sum;
        }
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::vector_linear_combination( const DataBlock<T>& vecA,const  DataBlock<T>& vecB, DataBlock<T> & vecC,const T CoeffA,const T CoeffB,const T CoeffC)
{
    const ptrdiff_t n=vecA.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            vecC(i)=CoeffC == T(0)?CoeffA*vecA(i)+CoeffB*vecB(i):CoeffC* vecC(i)+CoeffA*vecA(i)+CoeffB*vecB(i);
        }

    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            vecC(i)=CoeffC == T(0)?CoeffA*vecA(i)+CoeffB*vecB(i):CoeffC* vecC(i)+CoeffA*vecA(i)+CoeffB*vecB(i);
        }
    }
    else
    {
        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            vecC(i)=CoeffC == T(0)?CoeffA*vecA(i)+CoeffB*vecB(i):CoeffC* vecC(i)+CoeffA*vecA(i)+CoeffB*vecB(i);
        }
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::vector_linear_combination( const DataBlock<T>& vecA, DataBlock<T> & vecC,const T CoeffA,const T CoeffC)
{
    const ptrdiff_t n=vecA.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            vecC(i)=CoeffC == T(0)?CoeffA*vecA(i):CoeffC* vecC(i)+CoeffA*vecA(i);
        }

    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            vecC(i)=CoeffC == T(0)?CoeffA*vecA(i):CoeffC* vecC(i)+CoeffA*vecA(i);
        }
    }
    else
    {
        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            vecC(i)=CoeffC == T(0)?CoeffA*vecA(i):CoeffC* vecC(i)+CoeffA*vecA(i);
        }
    }

}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
T In_Kernel_Mathfunctions::vector_dot_product(const  DataBlock<T> &vec1, const DataBlock<T> &vec2)
{
    const ptrdiff_t n=vec1.dpextents[0];
    T result = T(0);
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {

        #pragma omp parallel for reduction(+:result)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            result += cond_conj( vec1(i)) * vec2(i);
        }
        return result;
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {

        #pragma omp simd reduction(+:result)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            result += cond_conj( vec1(i)) * vec2(i);
        }
    }
    else
    {
        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            result += cond_conj( vec1(i)) * vec2(i);
        }
    }

    return result;
}
#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
T In_Kernel_Mathfunctions::vector_dot_product_kahan(const DataBlock<T> &vec1, const DataBlock<T> &vec2)
{
    const ptrdiff_t n=vec1.dpextents[0];
    T result = T(0);
    T c_final = T(0);

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        const int total_threads = omp_get_max_threads();
        if (n < (ptrdiff_t)total_threads)
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
                    for (ptrdiff_t i = tid; i < n; i += actual_workers)
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
void In_Kernel_Mathfunctions::matrix_multiply_scalar(  const DataBlock<T>& M, const T alpha, DataBlock<T>& C)
{

    const ptrdiff_t n=C.dpextents[0];
    const ptrdiff_t m= C.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for (ptrdiff_t i = 0; i <n; ++i)
        {
            for (ptrdiff_t j = 0; j <  m; ++j)
            {

                C(i,j)= M(i, j) * alpha;
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for (ptrdiff_t i = 0; i <n; ++i)
        {
            for (ptrdiff_t j = 0; j <  m; ++j)
            {
                C(i,j)= M(i, j) * alpha;
            }
        }
    }
    else
    {

        for (ptrdiff_t i = 0; i <n; ++i)
        {
            #pragma omp unroll partial
            for (ptrdiff_t j = 0; j <  m; ++j)
            {
                C(i,j)= M(i, j) * alpha;
            }
        }
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::matrix_multiply_scalar(  DataBlock<T>& C, const T alpha)
{

    const ptrdiff_t n=C.dpextents[0];
    const ptrdiff_t m= C.dpextents[1];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd collapse(2)
        for (ptrdiff_t i = 0; i <n; ++i)
        {
            for (ptrdiff_t j = 0; j <  m; ++j)
            {
                C(i,j)*= alpha;
            }
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd collapse(2)
        for (ptrdiff_t i = 0; i <n; ++i)
        {
            for (ptrdiff_t j = 0; j <  m; ++j)
            {
                C(i,j)*= alpha;
            }
        }
    }
    else
    {

        for (ptrdiff_t i = 0; i <n; ++i)
        {
            #pragma omp unroll partial
            for (ptrdiff_t j = 0; j <  m; ++j)
            {
                C(i,j)*= alpha;
            }
        }
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::vector_multiply_scalar( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res)
{

    const ptrdiff_t n=vec.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            res(i) = vec(i)*scalar;
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            res(i) = vec(i)*scalar;
        }
    }
    else
    {
        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            res(i) = vec(i)*scalar;
        }
    }
}

#pragma omp end declare target




#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::vector_multiply_scalar( DataBlock<T>& vec,const T scalar)
{

    const ptrdiff_t n=vec.dpextents[0];
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp parallel for simd
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            vec(i) *= scalar;
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            vec(i) *= scalar;
        }
    }
    else
    {
        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            vec(i) *= scalar;
        }
    }
}

#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
T  In_Kernel_Mathfunctions::kahan_sum(const T *arr, ptrdiff_t n)
{
    T sum = T(0);
    T c = T(0);
    #pragma omp unroll partial
    for (ptrdiff_t i = 0; i < n; ++i)
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
T In_Kernel_Mathfunctions::neumaier_sum(const T* arr, ptrdiff_t n)
{

    if constexpr (is_complex<T>())
    {
        using ValueType = typename T::value_type;

        ValueType r_sum = ValueType(0);
        ValueType r_comp = ValueType(0);
        ValueType i_sum = ValueType(0);
        ValueType i_comp = ValueType(0);
        for (ptrdiff_t i = 0; i < n; ++i)
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
        for (ptrdiff_t i = 0; i < n; ++i)
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









#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::tensor_linear_combination(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,const T CoeffA,const T CoeffB, const T CoeffC)
{

    const ptrdiff_t rank=C.dprank;
    ptrdiff_t max_index=1;

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp simd reduction(*:max_index)
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp parallel for simd
        for (ptrdiff_t i = 0; i < max_index; ++i)
        {
            C(i) =CoeffC==0? CoeffA*A(i)+CoeffB*B(i): CoeffC*C(i)+CoeffA*A(i)+CoeffB*B(i);
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd reduction(*:max_index)
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp simd
        for (ptrdiff_t i = 0; i < max_index; ++i)
        {
            C(i) =CoeffC==0? CoeffA*A(i)+CoeffB*B(i): CoeffC*C(i)+CoeffA*A(i)+CoeffB*B(i);
        }
    }
    else
    {
        #pragma omp unroll partial
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i < max_index; ++i)
        {
            C(i) =CoeffC==0? CoeffA*A(i)+CoeffB*B(i): CoeffC*C(i)+CoeffA*A(i)+CoeffB*B(i);
        }
    }


}
#pragma omp end declare target





#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::tensor_linear_combination(const DataBlock<T>& A, DataBlock<T>& C,const T CoeffA, const T CoeffC)
{
    const ptrdiff_t rank=C.dprank;
    ptrdiff_t max_index=1;

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp simd reduction(*:max_index)
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp parallel for simd
        for (ptrdiff_t i = 0; i <max_index ; ++i)
        {
            C(i) =CoeffC==0? CoeffA*A(i): CoeffC*C(i)+CoeffA*A(i);
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd reduction(*:max_index)
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp simd
        for (ptrdiff_t i = 0; i <max_index ; ++i)
        {
            C(i) =CoeffC==0? CoeffA*A(i): CoeffC*C(i)+CoeffA*A(i);
        }
    }
    else
    {
        #pragma omp unroll partial
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i <max_index ; ++i)
        {
            C(i) =CoeffC==0? CoeffA*A(i): CoeffC*C(i)+CoeffA*A(i);
        }

    }


}
#pragma omp end declare target





#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::tensor_multiply_scalar(  const DataBlock<T>& M, const T alpha, DataBlock<T>& C)
{

    const ptrdiff_t rank=C.dprank;
    ptrdiff_t max_index=1;
    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp simd reduction(*:max_index)
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp parallel for simd
        for (ptrdiff_t i = 0; i <max_index; ++i)
        {
            C(i)= M(i) * alpha;
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd reduction(*:max_index)
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp simd
        for (ptrdiff_t i = 0; i <max_index; ++i)
        {
            C(i)= M(i) * alpha;
        }
    }
    else
    {
        #pragma omp unroll partial
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i <max_index; ++i)
        {
            C(i)= M(i) * alpha;
        }
    }

}
#pragma omp end declare target


#pragma omp begin declare target
template <OpenMPVariant Policy, typename T>
void In_Kernel_Mathfunctions::tensor_multiply_scalar(  DataBlock<T>& C, const T alpha)
{
    const ptrdiff_t rank=C.dprank;
    ptrdiff_t max_index=1;

    if constexpr (Policy == OpenMPVariant::ParallelSimd)
    {
        #pragma omp simd reduction(*:max_index)
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp parallel for simd
        for (ptrdiff_t i = 0; i <max_index; ++i)
        {
            C(i)= C(i) * alpha;
        }
    }
    else if constexpr (Policy == OpenMPVariant::Simd)
    {
        #pragma omp simd reduction(*:max_index)
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp simd
        for (ptrdiff_t i = 0; i <max_index; ++i)
        {
            C(i)= C(i) * alpha;
        }
    }
    else
    {
        #pragma omp unroll partial
        for(ptrdiff_t i=0; i<=rank; i++)
            max_index*=C.dpextents[i];

        #pragma omp unroll partial
        for (ptrdiff_t i = 0; i <max_index; ++i)
        {
            C(i)= C(i) * alpha;
        }
    }

}
#pragma omp end declare target



#endif
