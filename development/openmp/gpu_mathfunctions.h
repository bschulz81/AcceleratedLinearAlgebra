#ifndef GPUMATHFUNCTIONS
#define GPUMATHFUNCTIONS

#include "datablock.h"

#include "host_memory_functions.h"
#include "gpu_memory_functions.h"
#include "datablockutilities.h"




class GPU_Math_Functions
{
public:
    template <typename T>
    inline static void matrix_multiply_dot_g( const DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_dot_kahan_g( const DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_add_g(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_subtract_g( const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_dot_accumulate_g( const DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_dot_accumulate_kahan_g( const DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_add_accumulate_g( DataBlock<T>& A,const DataBlock<T>& B,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_subtract_accumulate_g(  DataBlock<T>& A,const  DataBlock<T>& B,int dev,bool update_host=true);

    template <typename T>
    inline static void matrix_multiply_scalar_g (const  DataBlock<T>& M, const T V, DataBlock<T>& C, int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_scalar_accumulate_g ( DataBlock<T>& M, const T V, int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_vector_g(const  DataBlock<T>&M, const DataBlock<T> &V, DataBlock<T>&C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_vector_g(const  DataBlock<T>&M, const T*V, DataBlock<T> &C, int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_vector_kahan_g(const  DataBlock<T>&M, const DataBlock<T> &V, DataBlock<T>& C,int dev,bool update_host=true);
    template <typename T>
    inline static void matrix_multiply_vector_kahan_g(const  DataBlock<T>&M, const T*V, DataBlock<T> & C, int dev,bool update_host=true);

    template <typename T>
    inline static void vector_multiply_scalar_g(const DataBlock<T>& vec, const T scalar,DataBlock<T>& res,int dev,bool update_host=true);
    template <typename T>
    inline static void vector_multiply_scalar_accumulate_g(DataBlock<T>& vec, const T scalar,int dev,bool update_host=true);
    template <typename T>
    inline static void vector_add_g(const  DataBlock<T>& vec1, const DataBlock<T>& vec2, DataBlock<T> & res,int dev,bool update_host=true);
    template <typename T>
    inline static void vector_add_accumulate_g(  DataBlock<T>& vec1, const DataBlock<T>& vec2,int dev,bool update_host=true);

    template <typename T>
    inline static void vector_subtract_g( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res,  int dev,bool update_host=true);
    template <typename T>
    inline static void vector_subtract_accumulate_g(  DataBlock<T>& vec1,const  DataBlock<T>& vec2,  int dev,bool update_host=true);


    template <typename T>
    inline static T dot_product_g( const DataBlock<T> &vec1,const  DataBlock<T> &vec2, int dev);
    template <typename T>
    inline static T dot_product_g_kahan( const DataBlock<T> &vec1,const  DataBlock<T> &vec2,int dev, int nteams, int nthreads_per_team );
    template <typename T>
    inline static void cholesky_decomposition_g(const DataBlock<T>& A, DataBlock<T> & L, int dev,bool update_host=true, bool initialize_output_to_zero=true);
    template <typename T>
    inline static void lu_decomposition_g(const DataBlock<T> &A,  DataBlock<T> & L,DataBlock<T> & U, int dev,bool update_host=true,bool initialize_output_to_zero=true);
    template <typename T>
    inline static void qr_decomposition_g(const DataBlock<T> &A,DataBlock<T>& Q, DataBlock<T> & R,  int dev,bool update_host=true,bool initialize_output_to_zero=true,bool memmaptempfiles=false);
    template <typename T>
    inline static void matrix_multiply_dot_sparse_g(const BlockedDataView<T>& Ablocks,const BlockedDataView<T>& Bblocks, DataBlock<T>& C,int dev,bool update_host=true,bool initialize_output_to_zero=true );
    template <typename T>
    inline static void matrix_multiply_dot_sparse_g(const BlockedDataView<T>& Ablocks,const DataBlock<T>& Bblocks, DataBlock<T>& C,int dev,bool update_host=true,bool initialize_output_to_zero=true );
    template <typename T>
    inline static void matrix_vector_multiply_sparse_g(const BlockedDataView<T>& A, const DataBlock<T>& x,          DataBlock<T>& y,int dev,bool update_host=true,bool initialize_output_to_zero=true ) ;
    template <typename T>
    inline static void matrix_vector_multiply_sparse_g(const BlockedDataView<T>& A, const BlockedDataView<T>& x,    DataBlock<T>& y,int dev,bool update_host=true,bool initialize_output_to_zero=true );

};


template <typename T>
void  GPU_Math_Functions::matrix_vector_multiply_sparse_g( const BlockedDataView<T>& A, const DataBlock<T>& x, DataBlock<T>& y,int dev,bool update_host,bool initialize_output_to_zero)
{
    const size_t mblocks = A.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];



    const size_t aext0 = A.dpextents[0];
    const size_t aext1 = A.dpextents[1];

    const size_t ystr0 = y.dpstrides[0];

    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadA(A, dev);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadx(x, dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloady(y, dev, true, update_host);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd device(dev)
        for(size_t i=0; i<y.dpextents[0]; i++)
            y.dpdata[i*ystr0]=T(0);
    }

    #pragma omp target teams distribute parallel for device(dev)
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
            #pragma omp simd reduction(+:sum)
            for (size_t kk = 0; kk < a_tile_cols; ++kk)
            {
                const size_t global_k = a_col_off + kk;
                sum += A(global_i,global_k) * x(global_k);
            }
            #pragma omp atomic update
            y(global_i)  +=sum;
        }

    }
}


template <typename T>
void GPU_Math_Functions::matrix_vector_multiply_sparse_g( const BlockedDataView<T>& A,  const BlockedDataView<T>& x,    DataBlock<T>& y,  int dev,bool update_host,bool initialize_output_to_zero)
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


    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadA(A, dev);
    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadx(x, dev);
    typename GPU_Memory_Functions::OffloadHelper<T> offloady(y, dev, true, update_host);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd device(dev)
        for(size_t i=0; i<y.dpextents[0]; i++)
            y.dpdata[i*ystr0]=T(0);
    }

    #pragma omp target teams distribute parallel for collapse(2)   device(dev)
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
                T sum =T(0);
                #pragma omp simd reduction(+:sum)
                for (size_t kk = k_start; kk < k_end; ++kk)
                {

                    sum += A(global_i,kk)* x(kk);
                }
                #pragma omp atomic update
                y(global_i ) += sum;
            }
        }
    }
}



template <typename T>
void GPU_Math_Functions::matrix_multiply_dot_sparse_g( const BlockedDataView<T>& A,  const DataBlock<T>& B, DataBlock<T>& C, int dev,bool update_host,bool initialize_output_to_zero)
{
    const size_t mblocks = A.usedblocks;

    const size_t Ablock_rows = A.block_shape[0];
    const size_t Ablock_cols = A.block_shape[1];


    const size_t Cstr0 = C.dpstrides[0];
    const size_t Cstr1 = C.dpstrides[1];

    const size_t aext0 = A.dpextents[0];
    const size_t aext1 = A.dpextents[1];
    const size_t bext0 = B.dpextents[0];
    const size_t bext1 = B.dpextents[1];

    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadA(A, dev);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadB(B, dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadC(C, dev, true, update_host);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd collapse(2)  device(dev)
        for(size_t i=0; i<C.dpextents[0]; i++)
        {
            for(size_t j=0; j<C.dpextents[1]; j++)
                C.dpdata[i*Cstr0+j*Cstr1]=T(0);
        }
    }

    #pragma omp target teams distribute parallel for device(dev)
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
                T sum=T(0);
                #pragma omp simd reduction(+:sum)
                for (size_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const size_t global_k = a_col_off + kk;



                    sum += A(global_i,global_k) * B(global_k,jj);
                }
                #pragma omp atomic update
                C(global_i, jj) +=sum;
            }
        }
    }
}




template <typename T>
void GPU_Math_Functions::matrix_multiply_dot_sparse_g( const BlockedDataView<T>& A,const BlockedDataView<T>& B,  DataBlock<T>& C, int dev,bool update_host,bool initialize_output_to_zero)
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



    const size_t aext0=A.dpextents[0];
    const size_t aext1=A.dpextents[1];

    const size_t bext0=B.dpextents[0];
    const size_t bext1=B.dpextents[1];

    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadA(A, dev);
    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadB(B, dev);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadC(C, dev, true, update_host);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(dev)
        for(size_t i=0; i<C.dpextents[0]; i++)
        {
            for(size_t j=0; j<C.dpextents[1]; j++)
                C.dpdata[i*str0+j*str1]=T(0);
        }
    }

    #pragma omp target teams distribute parallel for collapse(2) device(dev)
    for (size_t ia = 0; ia < mblocks; ++ia)
    {
        for (size_t jb = 0; jb < nblocks; ++jb)
        {
            const size_t a_start = A.pooled_offsets_starts[ia];
            const size_t* a_off =  A.pooled_offsets_flat + a_start;

            const size_t a_row_off = a_off[0];
            const size_t a_col_off = a_off[1];
            const  size_t a_rem_rows = aext0 - a_row_off;
            const  size_t a_rem_cols = aext1 - a_col_off;

            const size_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const size_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

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
                    #pragma omp simd reduction(+:sum)
                    for (size_t kk = k_start; kk < k_end; ++kk)
                    {
                        sum += A(global_i,kk) * B(kk,global_j);
                    }
                    #pragma omp atomic update
                    C(global_i,global_j) += sum;
                }
            }
        }
    }
}





template <typename T>
void GPU_Math_Functions::matrix_multiply_dot_g( const DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadA(A, dev, false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadB(B, dev, false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadC(C, dev, true, update_host);


    #pragma omp target teams distribute parallel for collapse(2)  device(dev)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = T(0);
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k)*B(k,j);
            }
            C(i,j)= sum;
        }
    }
}



template <typename T>
void GPU_Math_Functions::matrix_multiply_dot_kahan_g(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadA(A, dev, false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadB(B, dev, false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadC(C, dev, true, update_host);

    #pragma omp target teams distribute parallel for collapse(2) device(dev)
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

template <typename T>
void GPU_Math_Functions::matrix_add_g( const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host)
{

    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperA(A,dev,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperB(B,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd collapse(2)  device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }

}


template <typename T>
void GPU_Math_Functions::matrix_subtract_g( const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperA(A,dev,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperB(B,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd collapse(2)  device(dev)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
            C(i,j) =A(i,j)-B(i,j);
    }

}





template <typename T>
void GPU_Math_Functions::matrix_multiply_dot_accumulate_g( const DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadA(A, dev, false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadB(B, dev, false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadC(C, dev, false, update_host);


    #pragma omp target teams distribute parallel for collapse(2)  device(dev)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = T(0);
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k)*B(k,j);
            }
            C(i,j)+= sum;
        }
    }
}



template <typename T>
void GPU_Math_Functions::matrix_multiply_dot_accumulate_kahan_g(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadA(A, dev, false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadB(B, dev, false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadC(C, dev, false, update_host);

    #pragma omp target teams distribute parallel for collapse(2) device(dev)
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
            C(i,j)+= sum;
        }
    }


}

template <typename T>
void GPU_Math_Functions::matrix_add_accumulate_g(  DataBlock<T>& A,const DataBlock<T>& B,int dev,bool update_host)
{

    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperA(A,dev,false,update_host);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperB(B,dev,false);

    #pragma omp target teams distribute parallel for simd collapse(2)  device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j <m ; ++j)
        {
            A(i,j)+=B(i,j);
        }
    }

}


template <typename T>
void GPU_Math_Functions::matrix_subtract_accumulate_g(  DataBlock<T>& A,const  DataBlock<T>& B,int dev,bool update_host)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperA(A,dev,false,update_host);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperB(B,dev,false);

    #pragma omp target teams distribute parallel for simd collapse(2)  device(dev)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
            A(i,j)-=B(i,j);
    }

}




template <typename T>
void GPU_Math_Functions::matrix_multiply_vector_g( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T>& C,int dev,bool update_host)
{
    const size_t n= M.dpextents[0];
    const size_t m=V.dpextents[0];
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,dev,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperV(V,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,dev,true,update_host);
    #pragma omp target teams distribute parallel for device(dev)
    for (size_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        #pragma omp simd reduction(+: sum)
        for (size_t j = 0; j <m ; ++j)
        {
            sum+= M(i, j) * V(j);
        }
        C(i)=sum;
    }
}



template <typename T>
void GPU_Math_Functions::matrix_multiply_vector_kahan_g( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T>& C,int dev,bool update_host)
{


    const size_t n= M.dpextents[0];
    const size_t m=V.dpextents[0];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,dev,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperV(V,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for device(dev)
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


template <typename T>
void GPU_Math_Functions::matrix_multiply_vector_g( const DataBlock<T>&M, const T*V, DataBlock<T> & C,int dev,bool update_host)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    #pragma omp target enter data map (to:V[0:n])device(dev)
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for device(dev)
    for (size_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        #pragma omp simd reduction(+: sum)
        for (size_t j = 0; j <m ; ++j)
        {
            sum+= M(i, j) * V(j);
        }
        C(i)=sum;
    }

    #pragma omp target exit data map (release:V[0:n])device(dev)

}

template <typename T>
void GPU_Math_Functions::matrix_multiply_vector_kahan_g( const DataBlock<T>&M, const T*V, DataBlock<T> & C,int dev,bool update_host)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    #pragma omp target enter data map (to:V[0:n])device(dev)
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for  device(dev)
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

    #pragma omp target exit data map (release:V[0:n])device(dev)

}





template <typename T>
void GPU_Math_Functions::matrix_multiply_scalar_g( const  DataBlock<T>& M,const  T V, DataBlock<T>& C,int dev,bool update_host)
{

    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd collapse(2) device(dev)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V;
        }
    }


}



template <typename T>
void GPU_Math_Functions::matrix_multiply_scalar_accumulate_g( DataBlock<T>& M,const  T V,int dev,bool update_host)
{

    const size_t n=M.dpextents[0];
    const size_t m= M.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperM(M,dev,false,update_host);

    #pragma omp target teams distribute parallel for simd collapse(2) device(dev)
    for (size_t i = 0; i <n; ++i)
    {
        for (size_t j = 0; j <  m; ++j)
        {
            M(i, j) *= V;
        }
    }


}




template <typename T>
void GPU_Math_Functions::vector_multiply_scalar_g( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res,int dev,bool update_host)
{
    const size_t n=vec.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec(vec,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperres(res,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }


}


template <typename T>
void GPU_Math_Functions::vector_multiply_scalar_accumulate_g(  DataBlock<T>& vec,const T scalar,int dev,bool update_host)
{
    const size_t n=vec.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperres(vec,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        vec(i)*=scalar;
    }


}




template <typename T>
inline void GPU_Math_Functions::vector_add_g(const   DataBlock<T>& vec1, const DataBlock<T>& vec2, DataBlock<T> & res,int dev,bool update_host)
{
    const size_t n=vec1.dpextents[0];
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec1(vec1,dev,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec2(vec2,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperres(res,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}


template <typename T>
inline void GPU_Math_Functions::vector_subtract_g( const DataBlock<T>& vec1,const DataBlock<T>& vec2, DataBlock<T> & res,int dev,bool update_host)
{
    const size_t n=vec1.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec1(vec1,dev,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec2(vec2,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperres(res,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd  device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }



}


template <typename T>
inline void GPU_Math_Functions::vector_add_accumulate_g(   DataBlock<T>& vec1, const DataBlock<T>& vec2,int dev,bool update_host)
{
    const size_t n=vec1.dpextents[0];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelpervec1(vec1,dev,false,update_host);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec2(vec2,dev,false);

    #pragma omp target teams distribute parallel for simd device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        vec1(i)+=vec2(i);
    }

}


template <typename T>
inline void GPU_Math_Functions::vector_subtract_accumulate_g(  DataBlock<T>& vec1,const DataBlock<T>& vec2,int dev,bool update_host)
{
    const size_t n=vec1.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec1(vec1,dev,false,update_host);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec2(vec2,dev,false);

    #pragma omp target teams distribute parallel for simd  device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        vec1(i)-=vec2(i);
    }



}




template <typename T>
inline T GPU_Math_Functions::dot_product_g(const  DataBlock<T> &vec1, const DataBlock<T> &vec2,int dev)
{
    const size_t n=vec1.dpextents[0];

    T result=T(0);
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec1(vec1,dev,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec2(vec2,dev,false);


    if constexpr (is_complex<T>::value)
    {

        T result = T(0);
        #pragma omp target teams distribute parallel for simd map(tofrom:result)  reduction(+:result) device(dev)
        for (size_t i = 0; i < n; ++i)
        {
            T term = std::conj(vec1(i)) * vec2(i);
            result+=term;
        }

        return result;
    }


    else
    {
        #pragma omp target teams distribute parallel for simd map(tofrom:result) reduction(+:result) device(dev)
        for (size_t i = 0; i < n; ++i)
        {
            result += vec1(i) * vec2(i);
        }

        return result;
    }
}



template <typename T>
inline T GPU_Math_Functions::dot_product_g_kahan(const DataBlock<T> &vec1, const DataBlock<T> &vec2, int dev, int nteams, int nthreads_per_team)
{
    const size_t n = vec1.dpextents[0];
    const int total_threads = nteams * nthreads_per_team;

    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec1(vec1, dev, false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec2(dev, dev, false);

    if (n < (size_t)total_threads)
    {
        T result = T(0);

        #pragma omp target device(dev) map(tofrom: result)
        {
            T c_local = T(0);
            for (size_t i = 0; i < n; ++i)
            {
                T term;
                if constexpr (is_complex<T>::value)
                {
                    term = std::conj(vec1(i)) * vec2(i);
                }
                else
                {
                    term = vec1(i) * vec2(i);
                }

                T y = term - c_local;
                volatile T t = result + y;
                volatile T z = t - result;
                c_local = z - y;
                result = t;
            }
        }
        return result;
    }

    else
    {
        // Allocate raw target space
        T* thread_sums_dev = (T*)omp_target_alloc(sizeof(T) * total_threads, dev);
        T* thread_cs_dev   = (T*)omp_target_alloc(sizeof(T) * total_threads, dev);

        // Zero out the target buffers directly on the device using team mapping
        #pragma omp target teams distribute parallel for simd device(dev) is_device_ptr(thread_sums_dev, thread_cs_dev)
        for (int idx = 0; idx < total_threads; ++idx)
        {
            thread_sums_dev[idx] = T(0);
            thread_cs_dev[idx]   = T(0);
        }

        // Execute the parallel strided chunk loops across GPU thread groups
        #pragma omp target teams num_teams(nteams) thread_limit(nthreads_per_team) device(dev) is_device_ptr(thread_sums_dev, thread_cs_dev)
        {
            #pragma omp parallel
            {
                // Explicitly discover global thread workspace positioning across block barriers
                int tid = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num();

                if (tid < total_threads)
                {
                    T local_sum = T(0);
                    T c = T(0);

                    // Strided loop across global workspace
                    for (size_t i = tid; i < n; i += total_threads)
                    {
                        T term;
                        if constexpr (is_complex<T>::value)
                        {
                            term = std::conj(vec1(i)) * vec2(i);
                        }
                        else
                        {
                            term = vec1(i) * vec2(i);
                        }

                        T y = term - c;
                        volatile T t = local_sum + y;
                        volatile T z = t - local_sum;
                        c = z - y;
                        local_sum = t;
                    }

                    thread_sums_dev[tid] = local_sum;
                    thread_cs_dev[tid]   = c;
                }
            }
        }

        // Allocate local host memory to grab the partial chunks back
        T* host_sums=new T[total_threads];
        T* host_cs=new T[total_threads];

        // Copy chunk results back to the Host efficiently via device API pointers
        omp_target_memcpy(host_sums, thread_sums_dev, sizeof(T) * total_threads, 0, 0, omp_get_initial_device(), dev);
        omp_target_memcpy(host_cs, thread_cs_dev, sizeof(T) * total_threads, 0, 0, omp_get_initial_device(), dev);

        // Free device allocations safely
        omp_target_free(thread_sums_dev, dev);
        omp_target_free(thread_cs_dev, dev);

        // 3. Final Master Host Kahan Consolidation
        T result = T(0);
        T c_final = T(0);

        for (int tid = 0; tid < total_threads; ++tid)
        {
            // Process chunk accumulation
            T y1 = host_sums[tid] - c_final;
            volatile T t1 = result + y1;
            volatile T z1 = t1 - result;
            c_final = z1 - y1;
            result = t1;

            // Process associated chunk bit loss residual
            T y2 = host_cs[tid] - c_final;
            volatile T t2 = result + y2;
            volatile T z2 = t2 - result;
            c_final = z2 - y2;
            result = t2;
        }
        delete[]host_sums;
        delete[] host_cs;
        return result;
    }
}


template <typename T>
void GPU_Math_Functions::cholesky_decomposition_g(const DataBlock<T> & A,DataBlock<T> & L,int dev,bool update_host, bool initialize_output_to_zero)
{


    const size_t n = A.dpextents[0];

    L.pconjugate=false;

    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperA(A,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperL(L,dev,true,update_host);

    T* dataA=(T*)omp_get_mapped_ptr(A.dpdata,dev);
    T* dataL=(T*)omp_get_mapped_ptr(L.dpdata,dev);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(dev)
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=T(0);
            }
        }
    }

    for (size_t c = 0; c < n; ++c)
    {

        T tmp=T(0);
        #pragma omp target teams distribute  parallel for simd map(tofrom:tmp) reduction(+:tmp)  device(dev)
        for (size_t k = 0; k < c; ++k)
        {
            const T tmp3=L(c,k);
            tmp+= tmp3 * cond_conj( tmp3);
        }

        T tmp2;
        omp_target_memcpy(&tmp2,dataA,sizeof(T),0,sizeof(T)*(A.dpstrides[0]*c+A.dpstrides[1]*c),omp_get_initial_device(),dev);

        const T temp4=sqrt(tmp2-tmp);

        omp_target_memcpy(dataL,&temp4,sizeof(T),sizeof(T)*(L.dpstrides[0]*c+L.dpstrides[1]*c),0,dev,omp_get_initial_device());
        #pragma omp target teams distribute parallel for map(to:temp4) device(dev)
        for (size_t i = c + 1; i < n; ++i)
        {
            T tmp3 =T(0);
            #pragma omp simd reduction(+:tmp3)
            for (size_t k = 0; k < c; ++k)
            {
                tmp3 += L(i, k) * cond_conj( L(c, k));
            }
            tmp3=A(i, c)-tmp3;
            L(i, c)=tmp3/temp4;
        }
    }
}

template <typename T>
void GPU_Math_Functions::lu_decomposition_g(const DataBlock<T>& A, DataBlock<T> &L,DataBlock<T>& U,int dev, bool update_host,bool initialize_output_to_zero)
{
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperA(A,dev,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperL(L,dev,true,update_host);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperU(U,dev,true,update_host);

    size_t n = A.dpextents[0];
    L.pconjugate=false;
    U.pconjugate=false;
    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(dev)
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=T(0);
                U(i,j)=T(0);
            }
        }
    }

    T* udata=(T*)omp_get_mapped_ptr(U.dpdata,dev);
    size_t z=0;
    for (size_t c = 0; c < n; ++c)
    {
        #pragma omp target teams distribute parallel for device(dev)
        for (size_t i = c; i < n; ++i)
        {
            T temp=T(0);
            #pragma omp simd reduction(+:temp)
            for (size_t k = z; k < c; ++k)
            {
                temp += U( k,i) * L( c,k);
            }
            U(c,i)=A(c,i)-temp;
        }

        T temp4=T(0);
        omp_target_memcpy(&temp4,udata,sizeof(T),0,sizeof(T)*(U.dpstrides[0]*c+U.dpstrides[1]*c),omp_get_initial_device(),dev);

        #pragma omp target teams distribute parallel for  device(dev)
        for (size_t i = c; i < n; ++i)
        {
            T temp =T(0);
            #pragma omp simd reduction (+:temp)
            for (size_t k = z; k < c; ++k)
            {
                temp += U(k,c) * L( i,k);
            }
            temp=A(i,c)-temp;
            L(i,c)= temp/temp4;
        }
    }
}

template <typename T>
void GPU_Math_Functions::qr_decomposition_g(const DataBlock<T>& A, DataBlock<T>& Q, DataBlock<T>& R,  int dev,bool update_host,bool initialize_output_to_zero, bool memmap_tempfiles)
{


   int  step_size=(size_t)pow(A.dpextents[0],0.8385);

    if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    size_t n = A.dpextents[0];
    size_t m = A.dpextents[1];
    Q.pconjugate=false;
    R.pconjugate=false;

    bool aconj=A.pconjugate;
    // Initialize Q and R matrices
    size_t nm=n*m, mm=m*m;

        bool separate_device_memory=false;
#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif

        T * tempC;
        T * tempS;
        T*  tempM;
        if(separate_device_memory)
        {
            tempS= (T*) omp_target_alloc(sizeof(T)*nm, dev);
            tempC= (T*) omp_target_alloc(sizeof(T)*mm, dev);
            tempM= (T*) omp_target_alloc(sizeof(T)*A.dpdatalength, dev);
        }
        else
        {
            if(memmap_tempfiles)
            {
                tempS=Host_Memory_Functions::create_temp_mmap<T>(nm);
                tempC=Host_Memory_Functions::create_temp_mmap<T>(mm);
                tempM= Host_Memory_Functions::create_temp_mmap<T>(A.dpdatalength);
            }
            else
            {
            tempS= (T*)omp_alloc(sizeof(T)*nm,omp_default_mem_alloc);
            tempC= (T*) omp_alloc(sizeof(T)*mm, omp_default_mem_alloc);
            tempM= (T*) omp_alloc(sizeof(T)*A.dpdatalength, omp_default_mem_alloc);
            }
        }
        size_t aext[2]= {A.dpextents[0],A.dpextents[1]};
        size_t astr[2]= {A.dpstrides[0],A.dpstrides[1]};
        DataBlock<T> M(tempM,A.dpdatalength,A.dprowmajor,2,aext,astr,separate_device_memory,dev, false);



        DataBlock<T> tA=A,tQ=Q,tR=R;

        T* Mdptr=M.dpdata;

        if(separate_device_memory)
        {
            GPU_Memory_Functions::create_in(A,dev);
            GPU_Memory_Functions::create_out(Q,dev);
            GPU_Memory_Functions::create_out(R,dev);


            if(!A.dpdata_is_devptr)
                tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,dev);
            if(!Q.dpdata_is_devptr)
                tQ.dpdata=(T*) omp_get_mapped_ptr(Q.dpdata,dev);
            if(!R.dpdata_is_devptr)
                tR.dpdata=(T*) omp_get_mapped_ptr(R.dpdata,dev);

            tA.dpdata_is_devptr=true;
            tQ.dpdata_is_devptr=true;
            tR.dpdata_is_devptr=true;
            tA.devptr_devicenum=dev;
            tQ.devptr_devicenum=dev;
            tR.devptr_devicenum=dev;
        }

        const size_t Qstr0=Q.dpstrides[0];
        const size_t Qstr1=Q.dpstrides[1];
        const size_t Rstr0=R.dpstrides[0];
        const size_t Rstr1=R.dpstrides[1];
        const size_t Astr0=A.dpstrides[0];
        const size_t Astr1=A.dpstrides[1];
        T* tQdptr=tQ.dpdata;
        T* tRdptr=tR.dpdata;
        const T* tAdptr=tA.dpdata;
        if(initialize_output_to_zero)
        {

            #pragma omp target teams distribute parallel for simd collapse(2)is_device_ptr(tQdptr) device(dev)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    tQdptr[i*Qstr0 + j*Qstr1] = T(0);
                }
            }

            #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(tAdptr,tRdptr,Mdptr)device(dev)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < m; ++j)
                {
                   Mdptr[i*Astr0 + j*Astr1] =returnval(tAdptr[i*Astr0 + j*Astr1],aconj);
                   tRdptr[i*Rstr0 + j*Rstr1] = T(0);
                }
            }
        }
        else
        {
            #pragma omp target teams distribute parallel for simd collapse(2)  is_device_ptr(tAdptr,tRdptr,Mdptr) device(dev)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < m; ++j)
                {
                    Mdptr[i*Astr0+j*Astr1]=returnval(tAdptr[i*Astr0+j*Astr1],aconj);
                }
            }
        }

        size_t z = 0;

        for (size_t c = 0; c < m; ++c)
        {

            if (c == z +step_size)
            {

                size_t cz=c-z;
                size_t mc=m-c;
                // Extract submatrices

                size_t extBQ[2],strBQ[2];

                size_t extBM[2],strBM[2];

                DataBlock<T> BQ = DataBlockUtilities::matrix_subspan(tQ,0, z, n, cz,extBQ,strBQ);
                DataBlock<T> BM = DataBlockUtilities::matrix_subspan(M,0, c, n,mc,extBM,strBM);

                size_t tempCextt[2]= {cz,mc};
                size_t tempCstrt[2]= {mc,1};

                DataBlock<T>  C(tempC,cz*mc,true,2,tempCextt,tempCstrt,separate_device_memory,dev,false);


                size_t extBQT[2],strBQT[2];

                DataBlock<T> BQT=DataBlockUtilities::matrix_hermitian_transpose(BQ,extBQT,strBQT);

                GPU_Math_Functions::matrix_multiply_dot_g(BQT,BM,C,dev,false);



                size_t sextt[2]= {n,mc};
                size_t sstrt[2]= {mc,1};
                DataBlock<T>  S(tempS,n*mc,true,2,sextt,sstrt,separate_device_memory,dev, false);


                GPU_Math_Functions::matrix_multiply_dot_g(BQ,C,S,dev,false);


                T* Sdptr=S.dpdata;
                #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(Sdptr,Mdptr) device(dev)
                for (size_t i = 0; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        Mdptr[i*Astr0+j*Astr1] -= Sdptr[i*sstrt[0]+(j-c)*sstrt[1]];
                    }
                }
                z = c;
            }
//            // Extract column c of M

            size_t vext[1],vstr[1];
            DataBlock<T> v = DataBlockUtilities::matrix_column(M,c,vext,vstr);
            const size_t pextv0=vext[0];
            T* vdptr=v.dpdata;
            for (size_t j = z; j < c; ++j)
            {
                size_t uext[1],ustr[1];
                DataBlock<T>  u =DataBlockUtilities::matrix_column(tQ,j,uext,ustr);
                T*udptr=u.dpdata;
                T dot_pr=T(0);

                #pragma omp target teams distribute parallel for simd  map(tofrom: dot_pr) is_device_ptr(tQdptr,vdptr) reduction(+:dot_pr) device(dev)
                for (size_t i = 0; i < pextv0; ++i)
                {
                    dot_pr +=cond_conj( udptr[i*ustr[0]]) * vdptr[i*vstr[0]];
                }

                const T cdot_pr = dot_pr;
                #pragma omp target teams distribute parallel for simd is_device_ptr(udptr,vdptr)device(dev)
                for (size_t i = 0; i < pextv0; ++i)
                {
                    vdptr[i*vstr[0]] -= cdot_pr * udptr[i*ustr[0]];
                }

            }

            T norm = T(0);
            #pragma omp target  teams distribute parallel for simd map(tofrom:norm) is_device_ptr(vdptr)reduction(+:norm)device(dev)
            for (size_t i = 0; i < pextv0; ++i)
            {
                T val=vdptr[i*vstr[0]] ;
                norm += cond_conj(val) *vdptr[i*vstr[0]];
            }

            const T normc = sqrt(norm);

            #pragma omp target teams distribute parallel for simd is_device_ptr(tQdptr,vdptr) device(dev)
            for (size_t i = 0; i < pextv0; ++i)
            {
                tQdptr[i*Qstr0+c*Qstr1] = vdptr[i*vstr[0]]/normc;
            }

        }
        // Compute R = Q^T * A for real values and Q^\dagger for complex values... i have no algorithm for conjugate transpose multiplication...
        // the conjugate is done at best on the fly instead of making a separate copy... so make the conjugate transpose multiplication explicitely here.

        size_t extQT[2],strQT[2];
        DataBlock<T> QT=DataBlockUtilities::matrix_hermitian_transpose(tQ,extQT,strQT);

        GPU_Math_Functions::matrix_multiply_dot_g(QT,tA,tR,dev,false);

        if(separate_device_memory)
        {
            if(update_host)
            {
                GPU_Memory_Functions::update_host(Q,dev);
                GPU_Memory_Functions::update_host(R,dev);
            }
            GPU_Memory_Functions::release(A,dev);
            GPU_Memory_Functions::release(Q,dev);
            GPU_Memory_Functions::release(R,dev);

            omp_target_free(tempS, dev);
            omp_target_free(tempC, dev);
            omp_target_free(tempM, dev);
        }
        else
        {
            if(memmap_tempfiles)
            {
                Host_Memory_Functions::delete_temp_mmap<T>(tempS,nm);
                Host_Memory_Functions::delete_temp_mmap<T>(tempM,A.dpdatalength);
                Host_Memory_Functions::delete_temp_mmap<T>(tempC,mm);
            }
            else
            {
            omp_free(tempS, omp_default_mem_alloc);
            omp_free(tempC, omp_default_mem_alloc);
            omp_free(tempM, omp_default_mem_alloc);
            }
        }


    }




#endif

