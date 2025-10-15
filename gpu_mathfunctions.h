#ifndef GPUMATHFUNCTIONS
#define GPUMATHFUNCTIONS

#include "datablock.h"
#include "datablock_host_memory_functions.h"
#include "datablock_gpu_memory_functions.h"



template <typename T>
class GPU_Math_Functions
{
public:
    inline static void matrix_multiply_dot_g( const DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,int dev,bool update_host=true);
    inline static void matrix_multiply_dot_kahan_g( const DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,int dev,bool update_host=true);

    inline static void matrix_add_g(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host=true);
    inline static void matrix_subtract_g( const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host=true);

    inline static void matrix_multiply_vector_g(const  DataBlock<T>&M, const DataBlock<T> &V, DataBlock<T>&C,int dev,bool update_host=true);
    inline static void matrix_multiply_vector_g(const  DataBlock<T>&M, const T*V, DataBlock<T> &C, int dev,bool update_host=true);
    inline static void matrix_multiply_vector_kahan_g(const  DataBlock<T>&M, const DataBlock<T> &V, DataBlock<T>& C,int dev,bool update_host=true);
    inline static void matrix_multiply_vector_kahan_g(const  DataBlock<T>&M, const T*V, DataBlock<T> & C, int dev,bool update_host=true);

    inline static void matrix_multiply_scalar_g (const  DataBlock<T>& M, const T V, DataBlock<T>& C, int dev,bool update_host=true);
    inline static void vector_multiply_scalar_g(const DataBlock<T>& vec, const T scalar,DataBlock<T>& res,int dev,bool update_host=true);

    inline static void vector_add_g(const  DataBlock<T>& vec1, const DataBlock<T>& vec2, DataBlock<T> & res,int dev,bool update_host=true);
    inline static void vector_subtract_g( const DataBlock<T>& vec1,const  DataBlock<T>& vec2, DataBlock<T> & res,  int dev,bool update_host=true);

    inline static T dot_product_g( const DataBlock<T> &vec1,const  DataBlock<T> &vec2, int dev);
    inline static T dot_product_g_kahan( const DataBlock<T> &vec1,const  DataBlock<T> &vec2,int dev, int nteams, int nthreads_per_team );

    inline static void cholesky_decomposition_g(const DataBlock<T>& A, DataBlock<T> & L, int dev,bool update_host=true, bool initialize_output_to_zero=true);
    inline static void lu_decomposition_g(const DataBlock<T> &A,  DataBlock<T> & L,DataBlock<T> & U, int dev,bool update_host=true,bool initialize_output_to_zero=true);
    inline static void qr_decomposition_g(const DataBlock<T> &A,DataBlock<T>& Q, DataBlock<T> & R,  int dev,bool update_host=true,bool initialize_output_to_zero=true,bool memmaptempfiles=false);

    inline static void matrix_multiply_dot_sparse_g(const BlockedDataView<T>& Ablocks,const BlockedDataView<T>& Bblocks, DataBlock<T>& C,int dev,bool update_host=true,bool initialize_output_to_zero=true );
    inline static void matrix_multiply_dot_sparse_g(const BlockedDataView<T>& Ablocks,const DataBlock<T>& Bblocks, DataBlock<T>& C,int dev,bool update_host=true,bool initialize_output_to_zero=true );

    inline static void matrix_vector_multiply_sparse_g(const BlockedDataView<T>& A, const DataBlock<T>& x,          DataBlock<T>& y,int dev,bool update_host=true,bool initialize_output_to_zero=true ) ;
    inline static void matrix_vector_multiply_sparse_g(const BlockedDataView<T>& A, const BlockedDataView<T>& x,    DataBlock<T>& y,int dev,bool update_host=true,bool initialize_output_to_zero=true );

};


template <typename T>
void GPU_Math_Functions<T>::matrix_vector_multiply_sparse_g( const BlockedDataView<T>& A, const DataBlock<T>& x, DataBlock<T>& y,int dev,bool update_host,bool initialize_output_to_zero)
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

    typename DataBlock_GPU_Memory_Functions<T>::BlockedDataViewOffloadHelper offloadA(A, dev);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadx(x, dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloady(y, dev, true, update_host);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd device(dev)shared(y)
        for(size_t i=0; i<y.dpextents[0]; i++)
            y.dpdata[i*ystr0]=T(0);
    }

    #pragma omp target teams distribute parallel for shared(A,x,y)device(dev)
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
                const size_t a_index  = global_i * Astr0 + global_k * Astr1;
                const size_t x_index  = global_k * Xstr0;
                sum += A.dblock.dpdata[a_index] * x.dpdata[x_index];
            }
            #pragma omp atomic update
            y.dpdata[global_i * ystr0]  +=sum;
        }
    }
}


template <typename T>
void GPU_Math_Functions<T>::matrix_vector_multiply_sparse_g( const BlockedDataView<T>& A,  const BlockedDataView<T>& x,    DataBlock<T>& y,  int dev,bool update_host,bool initialize_output_to_zero)
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


    typename DataBlock_GPU_Memory_Functions<T>::BlockedDataViewOffloadHelper offloadA(A, dev);
    typename DataBlock_GPU_Memory_Functions<T>::BlockedDataViewOffloadHelper offloadx(x, dev);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloady(y, dev, true, update_host);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd device(dev)shared(y)
        for(size_t i=0; i<y.dpextents[0]; i++)
            y.dpdata[i*ystr0]=T(0);
    }

    #pragma omp target teams distribute  shared(A,x,y) device(dev)
    for (size_t ia = 0; ia < mblocks; ++ia)
    {
        #pragma omp parallel for shared(A,x,y)
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


template <typename T>
void GPU_Math_Functions<T>::matrix_multiply_dot_sparse_g( const BlockedDataView<T>& A,  const DataBlock<T>& B, DataBlock<T>& C, int dev,bool update_host,bool initialize_output_to_zero)
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
    const size_t bext0 = B.dpextents[0];
    const size_t bext1 = B.dpextents[1];

    typename DataBlock_GPU_Memory_Functions<T>::BlockedDataViewOffloadHelper offloadA(A, dev);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadB(B, dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadC(C, dev, true, update_host);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute shared(C) device(dev)
        for(size_t i=0; i<C.dpextents[0]; i++)
            #pragma omp parallel for simd  shared(C)
            for(size_t j=0; j<C.dpextents[1]; j++)
                C.dpdata[i*Cstr0+j*Cstr1]=T(0);
    }

    #pragma omp target teams distribute shared(A,B,C)device(dev)
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
            #pragma omp parallel for shared(A,B,C)
            for (size_t jj = 0; jj < bext1; ++jj)
            {
                T sum=T(0);
                #pragma omp simd reduction(+:sum)
                for (size_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const size_t global_k = a_col_off + kk;

                    const size_t a_index = global_i * Astr0 + global_k * Astr1;
                    const size_t b_index = global_k * Bstr0 + jj * Bstr1;

                    sum += A.dblock.dpdata[a_index] * B.dpdata[b_index];
                }
                #pragma omp atomic update
                C.dpdata[global_i * Cstr0 + jj * Cstr1] +=sum;
            }
        }
    }
}



template <typename T>
void GPU_Math_Functions<T>::matrix_multiply_dot_sparse_g( const BlockedDataView<T>& A,const BlockedDataView<T>& B,  DataBlock<T>& C, int dev,bool update_host,bool initialize_output_to_zero)
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

    const size_t Astr0=A.dblock.dpstrides[0];
    const size_t Astr1=A.dblock.dpstrides[1];
    const size_t Bstr0=B.dblock.dpstrides[0];
    const size_t Bstr1=B.dblock.dpstrides[1];

    const size_t aext0=A.dblock.dpextents[0];
    const size_t aext1=A.dblock.dpextents[1];

    const size_t bext0=B.dblock.dpextents[0];
    const size_t bext1=B.dblock.dpextents[1];

    typename DataBlock_GPU_Memory_Functions<T>::BlockedDataViewOffloadHelper offloadA(A, dev);
    typename DataBlock_GPU_Memory_Functions<T>::BlockedDataViewOffloadHelper offloadB(B, dev);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadC(C, dev, true, update_host);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute shared(C) device(dev)
        for(size_t i=0; i<C.dpextents[0]; i++)
            #pragma omp parallel for simd  shared(C)
            for(size_t j=0; j<C.dpextents[1]; j++)
                C.dpdata[i*str0+j*str1]=T(0);
    }

    #pragma omp  target teams distribute shared(A,B,C) device(dev)
    for (size_t ia = 0; ia < mblocks; ++ia)
    {
        #pragma omp parallel for shared(A,B,C)
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
                        const size_t a_index = (global_i * Astr0) +  (kk * Astr1);
                        const size_t b_index = (kk *Bstr0) +        (global_j *Bstr1);
                        sum += A.dblock.dpdata[a_index] * B.dblock.dpdata[b_index];
                    }
                    #pragma omp atomic update
                    C.dpdata[global_i*str0+global_j*str1] += sum;
                }
            }
        }
    }
}







template <typename T>
void GPU_Math_Functions<T>::matrix_multiply_dot_g( const DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadA(A, dev, false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadB(B, dev, false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadC(C, dev, true, update_host);

    const size_t Astr0=A.dpstrides[0];
    const size_t Astr1=A.dpstrides[1];
    const size_t Bstr0=B.dpstrides[0];
    const size_t Bstr1=B.dpstrides[1];
    const size_t Cstr0=C.dpstrides[0];
    const size_t Cstr1=C.dpstrides[1];

    #pragma omp target teams distribute shared(A,B,C) device(dev)
    for (size_t i = 0; i < rows; ++i)
        #pragma omp parallel for shared(A,B,C)
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = T(0);
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A.dpdata[i*Astr0+k*Astr1] *B.dpdata[k*Bstr0+j*Bstr1];
            }
            C.dpdata[i*Cstr0+j*Cstr1]= sum;
        }


}



template <typename T>
void GPU_Math_Functions<T>::matrix_multiply_dot_kahan_g(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadA(A, dev, false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadB(B, dev, false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadC(C, dev, true, update_host);

    #pragma omp target teams distribute shared(A,B,C) device(dev)
    for (size_t i = 0; i < rows; ++i)
        #pragma omp parallel for simd shared(A,B,C)
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
void GPU_Math_Functions<T>::matrix_add_g( const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host)
{

    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperA(A,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperB(B,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for shared(C,A,B) device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        #pragma omp simd
        for (size_t j = 0; j <m ; ++j)
        {
            C(i,j) =A(i,j)+B(i,j);
        }
    }


}


template <typename T>
void GPU_Math_Functions<T>::matrix_subtract_g( const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C,int dev,bool update_host)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperA(A,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperB(B,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for shared(C,A,B) device(dev)
    for (size_t i = 0; i <n; ++i)
    {
        #pragma omp simd
        for (size_t j = 0; j < m; ++j)
        {
            C(i,j) =A(i,j)-B(i,j);
        }
    }


}


template <typename T>
void GPU_Math_Functions<T>::matrix_multiply_vector_g( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T>& C,int dev,bool update_host)
{


    const size_t n= M.dpextents[0];
    const size_t m=V.dpextents[0];



    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperM(M,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperV(V,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for shared(M,V,C) device(dev)
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
void GPU_Math_Functions<T>::matrix_multiply_vector_kahan_g( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T>& C,int dev,bool update_host)
{


    const size_t n= M.dpextents[0];
    const size_t m=V.dpextents[0];



    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperM(M,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperV(V,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for shared(M,V,C) device(dev)
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
void GPU_Math_Functions<T>::matrix_multiply_vector_g( const DataBlock<T>&M, const T*V, DataBlock<T> & C,int dev,bool update_host)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    #pragma omp target enter data map (to:V[0:n])device(dev)
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperM(M,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for shared(M,V,C) device(dev)
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
void GPU_Math_Functions<T>::matrix_multiply_vector_kahan_g( const DataBlock<T>&M, const T*V, DataBlock<T> & C,int dev,bool update_host)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    #pragma omp target enter data map (to:V[0:n])device(dev)
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperM(M,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for shared(M,V,C) device(dev)
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
void GPU_Math_Functions<T>::matrix_multiply_scalar_g( const  DataBlock<T>& M,const  T V, DataBlock<T>& C,int dev,bool update_host)
{

    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperM(M,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for shared(C,M) device(dev)
    for (size_t i = 0; i <n; ++i)
    {
        #pragma omp simd
        for (size_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * V;
        }
    }


}



template <typename T>
void GPU_Math_Functions<T>::vector_multiply_scalar_g( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res,int dev,bool update_host)
{
    const size_t n=vec.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelpervec(vec,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperres(res,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd shared(res,vec) device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }


}




template <typename T>
inline void GPU_Math_Functions<T>::vector_add_g(const   DataBlock<T>& vec1, const DataBlock<T>& vec2, DataBlock<T> & res,int dev,bool update_host)
{
    const size_t n=vec1.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelpervec1(vec1,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelpervec2(vec2,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperres(res,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd shared(res,vec1,vec2) device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}


template <typename T>
inline void GPU_Math_Functions<T>::vector_subtract_g( const DataBlock<T>& vec1,const DataBlock<T>& vec2, DataBlock<T> & res,int dev,bool update_host)
{
    const size_t n=vec1.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelpervec1(vec1,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelpervec2(vec2,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperres(res,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd shared(vec1,vec2,res) device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }



}


template <typename T>
inline T GPU_Math_Functions<T>::dot_product_g(const  DataBlock<T> &vec1, const DataBlock<T> &vec2,int dev)
{
    const size_t n=vec1.dpextents[0];

    T result=T(0);



    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelpervec1(vec1,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelpervec2(vec2,dev,false);

    #pragma omp target teams distribute parallel for simd reduction(+:result)shared(vec1,vec2) device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }

    return result;
}


template <typename T>
inline T GPU_Math_Functions<T>::dot_product_g_kahan(const  DataBlock<T> &vec1, const DataBlock<T> &vec2,int dev, int nteams, int nthreads_per_team)
{
    const size_t n=vec1.dpextents[0];

    int total_threads = nteams * nthreads_per_team;

    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelpervec1(vec1,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelpervec2(vec2,dev,false);

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


        #pragma omp target teams distribute parallel for map(tofrom: thread_sums[0:total_threads], thread_cs[0:total_threads]) device(dev)
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


template <typename T>
void GPU_Math_Functions<T>::cholesky_decomposition_g(const DataBlock<T> & A,DataBlock<T> & L,int dev,bool update_host, bool initialize_output_to_zero)
{


    const size_t n = A.dpextents[0];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperA(A,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperL(L,dev,true,update_host);

    T* dataA=(T*)omp_get_mapped_ptr(A.dpdata,dev);
    T* dataL=(T*)omp_get_mapped_ptr(L.dpdata,dev);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute shared(L) device(dev)
        for (size_t i = 0; i < n; ++i)
            #pragma omp parallel for simd shared(L)
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=T(0);
            }
    }

    for (size_t c = 0; c < n; ++c)
    {

        T tmp=T(0);

        omp_target_memcpy(&tmp,dataA,sizeof(T),0,sizeof(T)*(A.dpstrides[0]*c+A.dpstrides[1]*c),omp_get_initial_device(),dev);


        #pragma omp target teams distribute  parallel for simd reduction(-:tmp) map(tofrom:tmp)shared(L) device(dev)
        for (size_t k = 0; k < c; ++k)
        {
            const T tmp3=L(c,k);
            tmp-= tmp3 * tmp3;
        }

        T temp4=sqrt(tmp);

        omp_target_memcpy(dataL,&temp4,sizeof(T),sizeof(T)*(L.dpstrides[0]*c+L.dpstrides[1]*c),0,dev,omp_get_initial_device());

        #pragma omp target teams distribute parallel for map(to:temp4) shared(A,L) device(dev)
        for (size_t i = c + 1; i < n; ++i)
        {
            T tmp2 = A(i, c);
            #pragma omp simd reduction(-:tmp2)
            for (size_t k = 0; k < c; ++k)
            {
                tmp2 -= L(i, k) * L(c, k);
            }
            L(i, c)=tmp2/temp4;
        }


    }



}

template <typename T>
void GPU_Math_Functions<T>::lu_decomposition_g(const DataBlock<T>& A, DataBlock<T> &L,DataBlock<T>& U,int dev, bool update_host,bool initialize_output_to_zero)
{


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperA(A,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperL(L,dev,true,update_host);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperU(U,dev,true,update_host);

    size_t n = A.dpextents[0];



    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute shared(U,L) device(dev)
        for (size_t i = 0; i < n; ++i)
            #pragma omp parallel for simd shared(U,L)
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=T(0);
                U(i,j)=T(0);
            }
    }

    T* udata=(T*)omp_get_mapped_ptr(U.dpdata,dev);

    size_t z=0;
    for (size_t c = 0; c < n; ++c)
    {
        #pragma omp target teams distribute shared(A,U,L) device(dev)
        for (size_t i = c; i < n; ++i)
        {
            T temp=A(c,i);
            #pragma omp parallel for simd reduction(-:temp) shared(L,U)
            for (size_t k = z; k < c; ++k)
            {
                temp -= U( k,i) * L( c,k);
            }
            U(c,i)=temp;
        }

        T temp4=T(0);
        omp_target_memcpy(&temp4,udata,sizeof(T),0,sizeof(T)*(U.dpstrides[0]*c+U.dpstrides[1]*c),omp_get_initial_device(),dev);

        #pragma omp target teams distribute shared(U,A,L) device(dev)
        for (size_t i = c; i < n; ++i)
        {
            T temp = A(i,c);
            #pragma omp parallel for simd reduction (-:temp)shared(U,L)
            for (size_t k = z; k < c; ++k)
            {
                temp -= U(k,c) * L( i,k);
            }
            L(i,c)=temp/temp4;
        }
    }


}
// Fast QR Decomposition Algorithm for mdspan
template <typename T>
void GPU_Math_Functions<T>::qr_decomposition_g(const DataBlock<T>& A, DataBlock<T>& Q, DataBlock<T>& R,  int dev,bool update_host,bool initialize_output_to_zero, bool memmap_tempfiles)
{
    const size_t n = A.dpextents[0];
    const size_t m = A.dpextents[1];

    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadhelperA(A,dev,false);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperQ(Q,dev,true,update_host);
    typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperR(R,dev,true,update_host);


    DataBlock<T> tQ=Q;

    if(!tQ.dpdata_is_devptr)
        tQ.dpdata=(T*) omp_get_mapped_ptr(Q.dpdata,dev);
    tQ.dpdata_is_devptr=true;


    DataBlock<T> M=DataBlock_GPU_Memory_Functions<T>::alloc_data_copy_strides_extents_device(A.dpdatalength,A.dprowmajor, A.dprank,A.dpextents,A.dpstrides,
                   memmap_tempfiles,dev);

    DataBlock_GPU_Memory_Functions<T>::create_in(M,dev);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute shared(tQ,M,A,R) device(dev)
        for (size_t i = 0; i < n; ++i)
        {
            #pragma omp parallel for simd  shared(tQ)
            for (size_t j = 0; j < n; ++j)
                tQ(i,j) = T(0);
            #pragma omp parallel for simd shared(M,A,R)
            for (size_t j = 0; j < m; ++j)
            {
                M(i,j)=A(i,j);
                R(i,j) = T(0);
            }
        }
    }
    else
    {
        #pragma omp target teams distribute shared(M,A) device(dev)
        for (size_t i = 0; i < n; ++i)
        {
            #pragma omp  parallel for simd shared(M,A)
            for (size_t j = 0; j < m; ++j)
            {
                M(i,j)=A(i,j);
            }
        }
    }

    size_t pext0=M.dpextents[0];
    for (size_t c = 0; c < m; ++c)
    {
        size_t pextv[1], pstrv[1];

        DataBlock<T> v = M.column(c, pextv, pstrv);  // current column, updated in place
        typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperv(v,dev,false,false);


        for (size_t j = 0; j < c; ++j)
        {
            size_t pextu[1], pstru[1];

            DataBlock<T> u = tQ.column(j, pextu, pstru);
            typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadhelperu(u,dev,false,false);
            T dot_pr = T(0);

            #pragma omp target teams distribute parallel for simd reduction(+:dot_pr)shared(u,v) device(dev)
            for (size_t i = 0; i < pext0; ++i)
                dot_pr += u(i) * v(i);

            const T cdot_pr = dot_pr;
            #pragma omp target teams distribute parallel for simd shared(u,v) device(dev)
            for (size_t i = 0; i < pext0; ++i)
                v(i) -= cdot_pr * u(i);
        }

        T norm = T(0);
        #pragma omp target teams distribute parallel for simd reduction(+:norm) shared(v)device(dev)
        for (size_t i = 0; i < pext0; ++i)
            norm += v(i) * v(i);

        const T normc = sqrt(norm);
        #pragma omp target teams distribute parallel for simd shared(v) device(dev)
        for (size_t i = 0; i < pext0; ++i)
            v(i) /= normc;

        #pragma omp target teams distribute parallel for simd shared(tQ,v) device(dev)
        for (size_t i = 0; i < pext0; ++i)
            tQ(i, c) = v(i);
    }



    const size_t rows = tQ.dpextents[0]; // Number of rows in A and C
    const size_t cols = A.dpextents[1]; // Number of columns in B and C
    const  size_t inner_dim = tQ.dpextents[1]; // Number of columns in A and rows in B

    #pragma omp target teams distribute shared(tQ,A,R) device(dev)
    for (size_t i = 0; i < rows; ++i)
    {
        #pragma omp parallel for shared(tQ,A,R)
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += tQ(k,i) *A(k,j);
            }
            R(i,j)= sum;
        }
    }
    DataBlock_GPU_Memory_Functions<T>::exit(M,dev);
    DataBlock_GPU_Memory_Functions<T>::free_copy_device(M,memmap_tempfiles,dev);
}




#endif

