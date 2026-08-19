#ifndef GPUMATHFUNCTIONShpp
#define GPUMATHFUNCTIONShpp





template <typename T>
void  GPU_Math_Functions::matrix_multiply_vector_sparse_g( const BlockedDataView<T>& A, const DataBlock<T>& x, DataBlock<T>& y,const T CoeffB,const T Coeffy,GPUOptions opt)
{
    const ptrdiff_t mblocks = A.usedblocks;

    const ptrdiff_t Ablock_rows = A.block_shape[0];
    const ptrdiff_t Ablock_cols = A.block_shape[1];



    const ptrdiff_t aext0 = A.dpextents[0];
    const ptrdiff_t aext1 = A.dpextents[1];

    const ptrdiff_t ystr0 = y.dpstrides[0];

    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadA(A, opt.device);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadx(x, opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloady(y, opt.device, false, opt.update_host);


        #pragma omp target teams distribute parallel for simd device(opt.device)
        for(ptrdiff_t i=0; i<y.dpextents[0]; i++)
        {
            const size_t index=i*ystr0;
            y.dpdata[index]=Coeffy==T(0)?T(0): Coeffy* y.dpdata[index];
        }




    #pragma omp target teams distribute parallel for device(opt.device)
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
            T sum =T(0) ;
            #pragma omp simd reduction(+:sum)
            for (ptrdiff_t kk = 0; kk < a_tile_cols; ++kk)
            {
                const ptrdiff_t global_k = a_col_off + kk;
                sum += A(global_i,global_k) * x(global_k);
            }
            #pragma omp atomic update
            y(global_i)  +=CoeffB*sum;
        }

    }
}


template <typename T>
void GPU_Math_Functions::matrix_multiply_vector_sparse_g( const BlockedDataView<T>& A,  const BlockedDataView<T>& x,    DataBlock<T>& y,  const T CoeffB,const T Coeffy,GPUOptions opt)
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


    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadA(A, opt.device);
    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadx(x, opt.device);
    typename GPU_Memory_Functions::OffloadHelper<T> offloady(y, opt.device, false, opt.update_host);


        #pragma omp target teams distribute parallel for simd device(opt.device)
        for(ptrdiff_t i=0; i<y.dpextents[0]; i++)
        {
            const size_t index=i*ystr0;
            y.dpdata[index]=Coeffy==T(0)?T(0): Coeffy* y.dpdata[index];
        }


    #pragma omp target teams distribute parallel for collapse(2)   device(opt.device)
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
                T sum =T(0);
                #pragma omp simd reduction(+:sum)
                for (ptrdiff_t kk = k_start; kk < k_end; ++kk)
                {

                    sum += A(global_i,kk)* x(kk);
                }
                #pragma omp atomic update
                y(global_i ) += CoeffB*sum;
            }
        }
    }
}



template <typename T>
void GPU_Math_Functions::matrix_multiply_dot_sparse_g( const BlockedDataView<T>& A,  const DataBlock<T>& B, DataBlock<T>& C,const T CoeffB,const T CoeffC,GPUOptions opt)
{
    const ptrdiff_t mblocks = A.usedblocks;

    const ptrdiff_t Ablock_rows = A.block_shape[0];
    const ptrdiff_t Ablock_cols = A.block_shape[1];


    const ptrdiff_t Cstr0 = C.dpstrides[0];
    const ptrdiff_t Cstr1 = C.dpstrides[1];

    const ptrdiff_t aext0 = A.dpextents[0];
    const ptrdiff_t aext1 = A.dpextents[1];
    const ptrdiff_t bext0 = B.dpextents[0];
    const ptrdiff_t bext1 = B.dpextents[1];

    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadA(A, opt.device);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadB(B, opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadC(C, opt.device, false, opt.update_host);


        #pragma omp target teams distribute parallel for simd collapse(2)  device(opt.device)
        for(ptrdiff_t i=0; i<C.dpextents[0]; i++)
        {
            for(ptrdiff_t j=0; j<C.dpextents[1]; j++)
            {
                const size_t index=i*Cstr0+j*Cstr1;
                C.dpdata[index]=CoeffC==T(0)?T(0): CoeffC*C.dpdata[index];
            }
        }


    #pragma omp target teams distribute parallel for device(opt.device)
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
                T sum=T(0);
                #pragma omp simd reduction(+:sum)
                for (ptrdiff_t kk = 0; kk < a_tile_cols; ++kk)
                {
                    const ptrdiff_t global_k = a_col_off + kk;



                    sum += A(global_i,global_k) * B(global_k,jj);
                }
                #pragma omp atomic update
                C(global_i, jj) +=CoeffB*sum;
            }
        }
    }
}




template <typename T>
void GPU_Math_Functions::matrix_multiply_dot_sparse_g( const BlockedDataView<T>& A,const BlockedDataView<T>& B,  DataBlock<T>& C, const T CoeffB,const T CoeffC,GPUOptions opt)
{
    // both A and B are assumed 2D
    const ptrdiff_t mblocks = A.usedblocks;
    const ptrdiff_t nblocks = B.usedblocks;

    const ptrdiff_t Ablock_rows = A.block_shape[0];
    const ptrdiff_t Ablock_cols = A.block_shape[1];
    const ptrdiff_t Bblock_rows = B.block_shape[0];
    const ptrdiff_t Bblock_cols = B.block_shape[1];

    const ptrdiff_t str0=C.dpstrides[0];
    const ptrdiff_t str1=C.dpstrides[1];



    const ptrdiff_t aext0=A.dpextents[0];
    const ptrdiff_t aext1=A.dpextents[1];

    const ptrdiff_t bext0=B.dpextents[0];
    const ptrdiff_t bext1=B.dpextents[1];

    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadA(A, opt.device);
    typename GPU_Memory_Functions::BlockedDataViewOffloadHelper<T> offloadB(B,opt.device);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadC(C, opt.device, false, opt.update_host);


        #pragma omp target teams distribute parallel for simd collapse(2) device(opt.device)
        for(ptrdiff_t i=0; i<C.dpextents[0]; i++)
        {
            for(ptrdiff_t j=0; j<C.dpextents[1]; j++)
            {
                const size_t index=i*str0+j*str1;
                C.dpdata[index]=CoeffC==T(0)?T(0): CoeffC*C.dpdata[index];
            }
        }


    #pragma omp target teams distribute parallel for collapse(2) device(opt.device)
    for (ptrdiff_t ia = 0; ia < mblocks; ++ia)
    {
        for (ptrdiff_t jb = 0; jb < nblocks; ++jb)
        {
            const ptrdiff_t a_start = A.pooled_offsets_starts[ia];
            const ptrdiff_t* a_off =  A.pooled_offsets_flat + a_start;

            const ptrdiff_t a_row_off = a_off[0];
            const ptrdiff_t a_col_off = a_off[1];
            const  ptrdiff_t a_rem_rows = aext0 - a_row_off;
            const  ptrdiff_t a_rem_cols = aext1 - a_col_off;

            const ptrdiff_t a_tile_rows = (Ablock_rows < a_rem_rows) ? Ablock_rows : a_rem_rows;
            const ptrdiff_t a_tile_cols = (Ablock_cols < a_rem_cols) ? Ablock_cols : a_rem_cols;

            const ptrdiff_t b_start = B.pooled_offsets_starts[jb];

            const ptrdiff_t* b_off = B.pooled_offsets_flat + b_start;
            const ptrdiff_t b_row_off = b_off[0];
            const ptrdiff_t b_col_off = b_off[1];

            const ptrdiff_t b_rem_rows = bext0 - b_row_off;
            const ptrdiff_t b_rem_cols = bext1 - b_col_off;

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
                    const ptrdiff_t global_j = b_col_off + jj;
                    T sum = T(0);
                    #pragma omp simd reduction(+:sum)
                    for (ptrdiff_t kk = k_start; kk < k_end; ++kk)
                    {
                        sum += A(global_i,kk) * B(kk,global_j);
                    }
                    #pragma omp atomic update
                    C(global_i,global_j) +=CoeffB* sum;
                }
            }
        }
    }
}





template <typename T>
void GPU_Math_Functions::matrix_multiply_dot_g( const DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,const T CoeffB,const T CoeffC,GPUOptions opt)
{
    const ptrdiff_t rows=A.dpextents[0];
    const ptrdiff_t cols=B.dpextents[1];
    const ptrdiff_t inner_dim=A.dpextents[1];

    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadA(A, opt.device, false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadB(B, opt.device, false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadC(C, opt.device, CoeffC==T(0), opt.update_host);


    #pragma omp target teams distribute parallel for collapse(2)  device(opt.device)
    for (ptrdiff_t i = 0; i < rows; ++i)
    {
        for (ptrdiff_t j = 0; j < cols; ++j)
        {
            T sum = T(0);
            #pragma omp simd reduction(+:sum)
            for (ptrdiff_t k = 0; k < inner_dim; ++k)
            {
                sum += A(i,k)*B(k,j);
            }
            C(i,j)=CoeffC==T(0)?CoeffB*sum:  CoeffC*C(i,j)+CoeffB*sum;
        }
    }
}



template <typename T>
void GPU_Math_Functions::matrix_multiply_dot_kahan_g(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,const T CoeffB,const T CoeffC,GPUOptions opt)
{
    const ptrdiff_t rows=A.dpextents[0];
    const ptrdiff_t cols=B.dpextents[1];
    const ptrdiff_t inner_dim=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadA(A, opt.device, false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadB(B, opt.device, false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadC(C, opt.device, CoeffC==T(0), opt.update_host);

    #pragma omp target teams distribute parallel for collapse(2) device(opt.device)
    for (ptrdiff_t i = 0; i < rows; ++i)
    {
        for (ptrdiff_t j = 0; j < cols; ++j)
        {
            T sum = T(0);
            T c=T(0);
            for (ptrdiff_t k = 0; k < inner_dim; ++k)
            {
                T y =  A(i,k) *B(k,j) - c;
                volatile T t = sum + y;
                volatile T z = t - sum;
                c = z - y;
                sum = t;
            }
             C(i,j)=CoeffC==T(0)?CoeffB*sum:  CoeffC*C(i,j)+CoeffB*sum;
        }
    }


}



template <typename T>
void GPU_Math_Functions::matrix_linear_combination_g( const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,const T CoeffA,const T CoeffB,const T CoeffC,GPUOptions opt)
{

    const ptrdiff_t n=A.dpextents[0];
    const ptrdiff_t m=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperA(A,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperB(B,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,opt.device,CoeffC==T(0),opt.update_host);

    #pragma omp target teams distribute parallel for simd collapse(2)  device(opt.device)
    for (ptrdiff_t i = 0; i < n; ++i)
    {
        for (ptrdiff_t j = 0; j <m ; ++j)
        {
            C(i,j) =CoeffC==T(0)?CoeffA*A(i,j)+CoeffB*B(i,j): CoeffC*C(i,j)+CoeffA*A(i,j)+CoeffB*B(i,j);
        }
    }

}

template <typename T>
void GPU_Math_Functions::matrix_linear_combination_g( const DataBlock<T>& A, DataBlock<T>& C,const T CoeffA,const T CoeffC,GPUOptions opt)
{

    const ptrdiff_t n=A.dpextents[0];
    const ptrdiff_t m=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperA(A,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,opt.device,CoeffC==T(0),opt.update_host);

    #pragma omp target teams distribute parallel for simd collapse(2)  device(opt.device)
    for (ptrdiff_t i = 0; i < n; ++i)
    {
        for (ptrdiff_t j = 0; j <m ; ++j)
        {
           C(i,j) =CoeffC==T(0)?CoeffA*A(i,j): CoeffC*C(i,j)+CoeffA*A(i,j);
        }
    }

}



template <typename T>
void GPU_Math_Functions::matrix_multiply_vector_g( const DataBlock<T>&M, const DataBlock<T>& V,DataBlock<T>&C,const T CoeffV,const T CoeffC,GPUOptions opt)
{
    const ptrdiff_t n= M.dpextents[0];
    const ptrdiff_t m=V.dpextents[0];
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperV(V,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,opt.device,CoeffC==T(0),opt.update_host);
    #pragma omp target teams distribute parallel for device(opt.device)
    for (ptrdiff_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        #pragma omp simd reduction(+: sum)
        for (ptrdiff_t j = 0; j <m ; ++j)
        {
            sum+= M(i, j) * V(j);
        }
        C(i)=CoeffC==T(0)?CoeffV*sum: CoeffC*C(i)+CoeffV*sum;
    }
}



template <typename T>
void GPU_Math_Functions::matrix_multiply_vector_kahan_g( const DataBlock<T>&M, const DataBlock<T>& V, DataBlock<T>&y, const T CoeffV,const T Coeffy,GPUOptions opt)
{


    const ptrdiff_t n= M.dpextents[0];
    const ptrdiff_t m=V.dpextents[0];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperV(V,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelpery(y,opt.device,Coeffy==T(0),opt.update_host);

    #pragma omp target teams distribute parallel for device(opt.device)
    for (ptrdiff_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        T c=T(0);
        for (ptrdiff_t j = 0; j <  m; ++j)
        {
            T y = M(i, j) * V(j) - c;
            volatile T t = sum + y;
            volatile T z = t - sum;
            c = z - y;
            sum = t;
        }
        y(i)=Coeffy==T(0)? CoeffV*sum: Coeffy*y(i)+CoeffV*sum;
    }


}


template <typename T>
void GPU_Math_Functions::matrix_multiply_vector_g( const DataBlock<T>&M, const T*V, DataBlock<T>&C,const T CoeffV,const T CoeffC,GPUOptions opt)
{


    const ptrdiff_t n= M.dpextents[0];
    const ptrdiff_t m=M.dpextents[1];

    #pragma omp target enter data map (to:V[0:n])device(opt.device)
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,opt.device,CoeffC==T(0),opt.update_host);

    #pragma omp target teams distribute parallel for device(opt.device)
    for (ptrdiff_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        #pragma omp simd reduction(+: sum)
        for (ptrdiff_t j = 0; j <m ; ++j)
        {
            sum+= M(i, j) * V(j);
        }
        C(i)=CoeffC==T(0)?CoeffV*sum: CoeffC*C(i)+CoeffV*sum;

    }

    #pragma omp target exit data map (release:V[0:n])device(opt.device)

}

template <typename T>
void GPU_Math_Functions::matrix_multiply_vector_kahan_g( const DataBlock<T>&M, const T*V, DataBlock<T>&C,const T CoeffV,const T CoeffC,GPUOptions opt)
{


    const ptrdiff_t n= M.dpextents[0];
    const ptrdiff_t m=M.dpextents[1];

    #pragma omp target enter data map (to:V[0:n])device(opt.device)
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,opt.device,CoeffC==T(0),opt.update_host);

    #pragma omp target teams distribute parallel for  device(opt.device)
    for (ptrdiff_t i = 0; i <n; ++i)
    {
        T sum=T(0);
        T c=T(0);
        for (ptrdiff_t j = 0; j <  m; ++j)
        {
            T y = M(i, j) * V[j] - c;
            volatile T t = sum + y;
            volatile T z = t - sum;
            c = z - y;
            sum = t;
        }
        C(i)=CoeffC==T(0)? CoeffV*sum: CoeffC*C(i)+CoeffV*sum;
    }

    #pragma omp target exit data map (release:V[0:n])device(opt.device)

}

template <typename T>
void GPU_Math_Functions::matrix_multiply_scalar_g( const  DataBlock<T>& M,const T alpha,DataBlock<T>&C,GPUOptions opt)
{

    const ptrdiff_t n=C.dpextents[0];
    const ptrdiff_t m= C.dpextents[1];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,opt.device,true,opt.update_host);

    #pragma omp target teams distribute parallel for simd collapse(2) device(opt.device)
    for (ptrdiff_t i = 0; i <n; ++i)
    {
        for (ptrdiff_t j = 0; j <  m; ++j)
        {
            C(i,j)= M(i, j) * alpha;
        }
    }


}



template <typename T>
void GPU_Math_Functions::matrix_multiply_scalar_g( DataBlock<T>& M,const  T scalar,GPUOptions opt)
{

    const ptrdiff_t n=M.dpextents[0];
    const ptrdiff_t m= M.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperM(M,opt.device,false,opt.update_host);

    #pragma omp target teams distribute parallel for simd collapse(2) device(opt.device)
    for (ptrdiff_t i = 0; i <n; ++i)
    {
        for (ptrdiff_t j = 0; j <  m; ++j)
        {
            M(i, j) *= scalar;
        }
    }


}




template <typename T>
void GPU_Math_Functions::vector_multiply_scalar_g( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res,GPUOptions opt)
{
    const ptrdiff_t n=vec.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec(vec,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperres(res,opt.device,true,opt.update_host);

    #pragma omp target teams distribute parallel for simd device(opt.device)
    for (ptrdiff_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }
}


template <typename T>
void GPU_Math_Functions::vector_multiply_scalar_g(  DataBlock<T>& vec,const T scalar,GPUOptions opt)
{
    const ptrdiff_t n=vec.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperres(vec,opt.device,false,opt.update_host);

    #pragma omp target teams distribute parallel for simd device(opt.device)
    for (ptrdiff_t i = 0; i < n; ++i)
    {
        vec(i)*=scalar;
    }


}




template <typename T>
inline void GPU_Math_Functions::vector_linear_combination_g(const   DataBlock<T>& vecA, const DataBlock<T>& vecB, DataBlock<T> & vecC,const T CoeffA,const T CoeffB,const T CoeffC,GPUOptions opt)
{
    const ptrdiff_t n=vecA.dpextents[0];
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec1(vecA,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec2(vecB,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperres(vecC,opt.device,CoeffC==T(0),opt.update_host);

    #pragma omp target teams distribute parallel for simd device(opt.device)
    for (ptrdiff_t i = 0; i < n; ++i)
    {
        vecC(i) =CoeffC==T(0)?CoeffA*vecA(i)+CoeffB*vecB(i): CoeffC*vecC(i)+CoeffA*vecA(i)+CoeffB*vecB(i);
    }

}




template <typename T>
inline void GPU_Math_Functions::vector_linear_combination_g(const   DataBlock<T>& vecA,  DataBlock<T> & vecC,const T CoeffA,const T CoeffC,GPUOptions opt)
{
    const ptrdiff_t n=vecA.dpextents[0];
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec1(vecA,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperres(vecC,opt.device,CoeffC==T(0),opt.update_host);

    #pragma omp target teams distribute parallel for simd device(opt.device)
    for (ptrdiff_t i = 0; i < n; ++i)
    {
        vecC(i) =CoeffC==T(0)?CoeffA*vecA(i): CoeffC*vecC(i)+CoeffA*vecA(i);
    }

}




template <typename T>
inline T GPU_Math_Functions::vector_dot_product_g(const  DataBlock<T> &vec1, const DataBlock<T> &vec2,GPUOptions opt)
{
    const ptrdiff_t n=vec1.dpextents[0];

    T result=T(0);
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec1(vec1,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec2(vec2,opt.device,false);


    if constexpr (is_complex<T>::value)
    {

        T result = T(0);
        #pragma omp target teams distribute parallel for simd map(tofrom:result)  reduction(+:result) device(opt.device)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            T term = std::conj(vec1(i)) * vec2(i);
            result+=term;
        }

        return result;
    }


    else
    {
        #pragma omp target teams distribute parallel for simd map(tofrom:result) reduction(+:result) device(opt.device)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            result += vec1(i) * vec2(i);
        }

        return result;
    }
}

template<typename T>
inline T GPU_Math_Functions::vector_dot_product_kahan_g(const DataBlock<T>& vec1,const DataBlock<T>& vec2, GPUOptions options)
{
    DeviceInfo info =query_device_team_thread_counts(options.device);

    const ptrdiff_t n = vec1.dpextents[0];
    const int total_threads = info.num_teams * info.threads_per_team;

    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec1(vec1, info.dev_id, false);
    typename  GPU_Memory_Functions::OffloadHelperConst<T> offloadhelpervec2(vec2, info.dev_id, false);

    if (n < (ptrdiff_t)total_threads)
    {
        T result = T(0);
        #pragma omp target device(info.dev_id) map(tofrom: result)
        {
            T c_local = T(0);
            for (ptrdiff_t i = 0; i < n; ++i)
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
        T* thread_sums_dev = (T*)omp_target_alloc(sizeof(T) * total_threads, info.dev_id);
        T* thread_cs_dev   = (T*)omp_target_alloc(sizeof(T) * total_threads, info.dev_id);

        #pragma omp target teams distribute parallel for simd device(info.dev_id) is_device_ptr(thread_sums_dev, thread_cs_dev)
        for (int idx = 0; idx < total_threads; ++idx)
        {
            thread_sums_dev[idx] = T(0);
            thread_cs_dev[idx]   = T(0);
        }
        #pragma omp target teams num_teams(info.num_teams) thread_limit(info.threads_per_team) device(info.dev_id) is_device_ptr(thread_sums_dev, thread_cs_dev)
        {
            #pragma omp parallel
            {

                int tid = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num();

                if (tid < total_threads)
                {
                    T local_sum = T(0);
                    T c = T(0);


                    for (ptrdiff_t i = tid; i < n; i += total_threads)
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


        T* host_sums=new T[total_threads];
        T* host_cs=new T[total_threads];

        omp_target_memcpy(host_sums, thread_sums_dev, sizeof(T) * total_threads, 0, 0, omp_get_initial_device(), info.dev_id);
        omp_target_memcpy(host_cs, thread_cs_dev, sizeof(T) * total_threads, 0, 0, omp_get_initial_device(), info.dev_id);


        omp_target_free(thread_sums_dev, info.dev_id);
        omp_target_free(thread_cs_dev, info.dev_id);

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
void GPU_Math_Functions::cholesky_decomposition_g(const DataBlock<T> & A,DataBlock<T> & L,bool initialize_output_to_zero,GPUOptions opt)
{


    const ptrdiff_t n = A.dpextents[0];

    L.dpconjugate=false;

    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperA(A,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperL(L,opt.device,true,opt.update_host);

    T* dataA=(T*)omp_get_mapped_ptr(A.dpdata,opt.device);
    T* dataL=(T*)omp_get_mapped_ptr(L.dpdata,opt.device);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(opt.device)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            for (ptrdiff_t j = 0; j <n; ++j)
            {
                L(i,j)=T(0);
            }
        }
    }

    for (ptrdiff_t c = 0; c < n; ++c)
    {

        T tmp=T(0);
        #pragma omp target teams distribute  parallel for simd map(tofrom:tmp) reduction(+:tmp)  device(opt.device)
        for (ptrdiff_t k = 0; k < c; ++k)
        {
            const T tmp3=L(c,k);
            tmp+= tmp3 * cond_conj( tmp3);
        }

        T tmp2;
        omp_target_memcpy(&tmp2,dataA,sizeof(T),0,sizeof(T)*(A.dpstrides[0]*c+A.dpstrides[1]*c),omp_get_initial_device(),opt.device);

        const T temp4=sqrt(tmp2-tmp);

        omp_target_memcpy(dataL,&temp4,sizeof(T),sizeof(T)*(L.dpstrides[0]*c+L.dpstrides[1]*c),0,opt.device,omp_get_initial_device());
        #pragma omp target teams distribute parallel for map(to:temp4) device(opt.device)
        for (ptrdiff_t i = c + 1; i < n; ++i)
        {
            T tmp3 =T(0);
            #pragma omp simd reduction(+:tmp3)
            for (ptrdiff_t k = 0; k < c; ++k)
            {
                tmp3 += L(i, k) * cond_conj( L(c, k));
            }
            tmp3=A(i, c)-tmp3;
            L(i, c)=tmp3/temp4;
        }
    }
}

template <typename T>
void GPU_Math_Functions::lu_decomposition_g(const DataBlock<T>& A, DataBlock<T> &L,DataBlock<T>& U,  bool initialize_output_to_zero,GPUOptions opt)
{
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperA(A,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperL(L,opt.device,true,opt.update_host);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperU(U,opt.device,true,opt.update_host);

    ptrdiff_t n = A.dpextents[0];
    L.dpconjugate=false;
    U.dpconjugate=false;
    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(opt.device)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            for (ptrdiff_t j = 0; j <n; ++j)
            {
                L(i,j)=T(0);
                U(i,j)=T(0);
            }
        }
    }

    T* udata=(T*)omp_get_mapped_ptr(U.dpdata,opt.device);
    ptrdiff_t z=0;
    for (ptrdiff_t c = 0; c < n; ++c)
    {
        #pragma omp target teams distribute parallel for device(opt.device)
        for (ptrdiff_t i = c; i < n; ++i)
        {
            T temp=T(0);
            #pragma omp simd reduction(+:temp)
            for (ptrdiff_t k = z; k < c; ++k)
            {
                temp += U( k,i) * L( c,k);
            }
            U(c,i)=A(c,i)-temp;
        }

        T temp4=T(0);
        omp_target_memcpy(&temp4,udata,sizeof(T),0,sizeof(T)*(U.dpstrides[0]*c+U.dpstrides[1]*c),omp_get_initial_device(),opt.device);

        #pragma omp target teams distribute parallel for  device(opt.device)
        for (ptrdiff_t i = c; i < n; ++i)
        {
            T temp =T(0);
            #pragma omp simd reduction (+:temp)
            for (ptrdiff_t k = z; k < c; ++k)
            {
                temp += U(k,c) * L( i,k);
            }
            temp=A(i,c)-temp;
            L(i,c)= temp/temp4;
        }
    }
}

template <typename T>
void GPU_Math_Functions::qr_decomposition_g(const DataBlock<T>& A, DataBlock<T>& Q, DataBlock<T>& R,  bool initialize_output_to_zero, bool memmap_tempfiles,GPUOptions opt)
{


    int  step_size=(ptrdiff_t)pow(A.dpextents[0],0.8385);

    if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    ptrdiff_t n = A.dpextents[0];
    ptrdiff_t m = A.dpextents[1];
    Q.dpconjugate=false;
    R.dpconjugate=false;

    bool aconj=A.dpconjugate;
    // Initialize Q and R matrices
    ptrdiff_t nm=n*m, mm=m*m;

    bool separate_device_memory=false;
#if !defined(Unified_Shared_Memory)
    separate_device_memory=true;
#endif

    T * tempC;
    T * tempS;
    T*  tempM;
    if(separate_device_memory)
    {
        tempS= (T*) omp_target_alloc(sizeof(T)*nm, opt.device);
        tempC= (T*) omp_target_alloc(sizeof(T)*mm, opt.device);
        tempM= (T*) omp_target_alloc(sizeof(T)*A.dpdatalength, opt.device);
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
    ptrdiff_t aext[2]= {A.dpextents[0],A.dpextents[1]};
    ptrdiff_t astr[2]= {A.dpstrides[0],A.dpstrides[1]};
    DataBlockConfig aconf({.dprowmajor=A.dpconfig.dprowmajor,
                           .pmemmap=memmap_tempfiles,
                           .data_is_devptr=separate_device_memory,
                           .devicenum=opt.device,
                          });
    DataBlock<T> M(tempM,A.dpdatalength,2,aext,astr,aconf);



    DataBlock<T> tA=A,tQ=Q,tR=R;

    T* Mdptr=M.dpdata;

    if(separate_device_memory)
    {
        GPU_Memory_Functions::create_in(A,opt.device);
        GPU_Memory_Functions::create_out(Q,opt.device);
        GPU_Memory_Functions::create_out(R,opt.device);


        if(!A.dpconfig.data_is_devptr)
            tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,opt.device);
        if(!Q.dpconfig.data_is_devptr)
            tQ.dpdata=(T*) omp_get_mapped_ptr(Q.dpdata,opt.device);
        if(!R.dpconfig.data_is_devptr)
            tR.dpdata=(T*) omp_get_mapped_ptr(R.dpdata,opt.device);

        tA.dpconfig.data_is_devptr=true;
        tQ.dpconfig.data_is_devptr=true;
        tR.dpconfig.data_is_devptr=true;
        tA.dpconfig.devicenum=opt.device;
        tQ.dpconfig.devicenum=opt.device;
        tR.dpconfig.devicenum=opt.device;
    }

    const ptrdiff_t Qstr0=Q.dpstrides[0];
    const ptrdiff_t Qstr1=Q.dpstrides[1];
    const ptrdiff_t Rstr0=R.dpstrides[0];
    const ptrdiff_t Rstr1=R.dpstrides[1];
    const ptrdiff_t Astr0=A.dpstrides[0];
    const ptrdiff_t Astr1=A.dpstrides[1];
    T* tQdptr=tQ.dpdata;
    T* tRdptr=tR.dpdata;
    const T* tAdptr=tA.dpdata;
    if(initialize_output_to_zero)
    {

        #pragma omp target teams distribute parallel for simd collapse(2)is_device_ptr(tQdptr) device(opt.device)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            for (ptrdiff_t j = 0; j < n; ++j)
            {
                tQdptr[i*Qstr0 + j*Qstr1] = T(0);
            }
        }

        #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(tAdptr,tRdptr,Mdptr)device(opt.device)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            for (ptrdiff_t j = 0; j < m; ++j)
            {
                Mdptr[i*Astr0 + j*Astr1] =returnval(tAdptr[i*Astr0 + j*Astr1],aconj);
                tRdptr[i*Rstr0 + j*Rstr1] = T(0);
            }
        }
    }
    else
    {
        #pragma omp target teams distribute parallel for simd collapse(2)  is_device_ptr(tAdptr,tRdptr,Mdptr) device(opt.device)
        for (ptrdiff_t i = 0; i < n; ++i)
        {
            for (ptrdiff_t j = 0; j < m; ++j)
            {
                Mdptr[i*Astr0+j*Astr1]=returnval(tAdptr[i*Astr0+j*Astr1],aconj);
            }
        }
    }

    ptrdiff_t z = 0;
    DataBlockConfig cconf({.dprowmajor=true,
                           .pmemmap=memmap_tempfiles,
                           .data_is_devptr=separate_device_memory,
                           .devicenum=opt.device
                          });
    for (ptrdiff_t c = 0; c < m; ++c)
    {

        if (c == z +step_size)
        {

            ptrdiff_t cz=c-z;
            ptrdiff_t mc=m-c;
            // Extract submatrices

            ptrdiff_t extBQ[2];
            ptrdiff_t strBQ[2];

            ptrdiff_t extBM[2];
            ptrdiff_t strBM[2];

            DataBlock<T> BQ = DataBlockUtilities::matrix_subspan(tQ,0, z, n, cz,extBQ,strBQ);
            DataBlock<T> BM = DataBlockUtilities::matrix_subspan(M,0, c, n,mc,extBM,strBM);

            ptrdiff_t tempCextt[2]= {cz,mc};
            ptrdiff_t tempCstrt[2]= {mc,1};

            DataBlock<T>  C(tempC,cz*mc,2,tempCextt,tempCstrt,cconf);


            ptrdiff_t extBQT[2];
            ptrdiff_t strBQT[2];

            DataBlock<T> BQT=DataBlockUtilities::matrix_hermitian_transpose(BQ,extBQT,strBQT);

            GPU_Math_Functions::matrix_multiply_dot_g(BQT,BM,C,GPUOptions{.device=opt.device,.update_host=false});



            ptrdiff_t sextt[2]= {n,mc};
            ptrdiff_t sstrt[2]= {mc,1};
            DataBlock<T>  S(tempS,n*mc,2,sextt,sstrt,cconf);


            GPU_Math_Functions::matrix_multiply_dot_g(BQ,C,S,GPUOptions{.device=opt.device,.update_host=false});


            T* Sdptr=S.dpdata;
            #pragma omp target teams distribute parallel for simd collapse(2) is_device_ptr(Sdptr,Mdptr) device(opt.device)
            for (ptrdiff_t i = 0; i < n; ++i)
            {
                for (ptrdiff_t j = c; j < n; ++j)
                {
                    Mdptr[i*Astr0+j*Astr1] -= Sdptr[i*sstrt[0]+(j-c)*sstrt[1]];
                }
            }
            z = c;
        }
//            // Extract column c of M

        ptrdiff_t vext[1];
        ptrdiff_t vstr[1];
        DataBlock<T> v = DataBlockUtilities::matrix_column(M,c,vext,vstr);
        const ptrdiff_t pextv0=vext[0];
        T* vdptr=v.dpdata;
        for (ptrdiff_t j = z; j < c; ++j)
        {
            ptrdiff_t uext[1];
            ptrdiff_t ustr[1];
            DataBlock<T>  u =DataBlockUtilities::matrix_column(tQ,j,uext,ustr);
            T*udptr=u.dpdata;
            T dot_pr=T(0);

            #pragma omp target teams distribute parallel for simd  map(tofrom: dot_pr) is_device_ptr(tQdptr,vdptr) reduction(+:dot_pr) device(opt.device)
            for (ptrdiff_t i = 0; i < pextv0; ++i)
            {
                dot_pr +=cond_conj( udptr[i*ustr[0]]) * vdptr[i*vstr[0]];
            }

            const T cdot_pr = dot_pr;
            #pragma omp target teams distribute parallel for simd is_device_ptr(udptr,vdptr)device(opt.device)
            for (ptrdiff_t i = 0; i < pextv0; ++i)
            {
                vdptr[i*vstr[0]] -= cdot_pr * udptr[i*ustr[0]];
            }

        }

        T norm = T(0);
        #pragma omp target  teams distribute parallel for simd map(tofrom:norm) is_device_ptr(vdptr)reduction(+:norm)device(opt.device)
        for (ptrdiff_t i = 0; i < pextv0; ++i)
        {
            T val=vdptr[i*vstr[0]] ;
            norm += cond_conj(val) *vdptr[i*vstr[0]];
        }

        const T normc = sqrt(norm);

        #pragma omp target teams distribute parallel for simd is_device_ptr(tQdptr,vdptr) device(opt.device)
        for (ptrdiff_t i = 0; i < pextv0; ++i)
        {
            tQdptr[i*Qstr0+c*Qstr1] = vdptr[i*vstr[0]]/normc;
        }

    }

    ptrdiff_t extQT[2];
    ptrdiff_t strQT[2];
    DataBlock<T> QT=DataBlockUtilities::matrix_hermitian_transpose(tQ,extQT,strQT);

    GPU_Math_Functions::matrix_multiply_dot_g(QT,tA,tR,GPUOptions{.device=opt.device,.update_host=false});

    if(separate_device_memory)
    {
        if(opt.update_host)
        {
            GPU_Memory_Functions::update_host(Q,opt.device);
            GPU_Memory_Functions::update_host(R,opt.device);
        }
        GPU_Memory_Functions::release(A,opt.device);
        GPU_Memory_Functions::release(Q,opt.device);
        GPU_Memory_Functions::release(R,opt.device);

        omp_target_free(tempS, opt.device);
        omp_target_free(tempC, opt.device);
        omp_target_free(tempM, opt.device);
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



template <typename T>
void GPU_Math_Functions::tensor_linear_combination_g( const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,const T CoeffA,const T CoeffB,const T CoeffC,GPUOptions opt)
{

    const ptrdiff_t rank=C.dprank;
    ptrdiff_t max_index=1;

    #pragma omp simd reduction(*:max_index)
    for(ptrdiff_t i=0; i<=rank; i++)
        max_index*=C.dpextents[i];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperA(A,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperB(B,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,opt.device,CoeffC==T(0),opt.update_host);

    #pragma omp target teams distribute parallel for simd  device(opt.device)
    for (ptrdiff_t i = 0; i <max_index ; ++i)
    {
        C(i) =CoeffC==T(0)?CoeffA*A(i)+CoeffB*B(i): CoeffC*C(i)+CoeffA*A(i)+CoeffB*B(i);
    }


}

template <typename T>
void GPU_Math_Functions::tensor_linear_combination_g( const DataBlock<T>& A, DataBlock<T>& C,const T CoeffA,const T CoeffC,GPUOptions opt)
{

const ptrdiff_t rank=C.dprank;
    ptrdiff_t max_index=1;

    #pragma omp simd reduction(*:max_index)
    for(ptrdiff_t i=0; i<=rank; i++)
        max_index*=C.dpextents[i];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperA(A,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,opt.device,CoeffC==T(0),opt.update_host);

    #pragma omp target teams distribute parallel for device(opt.device)
    for (ptrdiff_t i = 0; i < max_index; ++i)
    {
        C(i) =CoeffC==T(0)?CoeffA*A(i): CoeffC*C(i)+CoeffA*A(i);
    }

}


template <typename T>
void GPU_Math_Functions::tensor_multiply_scalar_g( const  DataBlock<T>& M,const T alpha,DataBlock<T>&C,GPUOptions opt)
{

 const ptrdiff_t rank=C.dprank;
    ptrdiff_t max_index=1;

    #pragma omp simd reduction(*:max_index)
    for(ptrdiff_t i=0; i<=rank; i++)
        max_index*=C.dpextents[i];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelperConst<T> offloadhelperM(M,opt.device,false);
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperC(C,opt.device,true,opt.update_host);

    #pragma omp target teams distribute parallel for simd device(opt.device)
    for (ptrdiff_t i = 0; i <max_index; ++i)
    {
        C(i)= M(i) * alpha;
    }


}



template <typename T>
void GPU_Math_Functions::tensor_multiply_scalar_g( DataBlock<T>& M,const  T scalar,GPUOptions opt)
{
const ptrdiff_t rank=M.dprank;
    ptrdiff_t max_index=1;

    #pragma omp simd reduction(*:max_index)
    for(ptrdiff_t i=0; i<=rank; i++)
        max_index*=M.dpextents[i];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename GPU_Memory_Functions::OffloadHelper<T> offloadhelperM(M,opt.device,false,opt.update_host);

    #pragma omp target teams distribute parallel for simd device(opt.device)
    for (ptrdiff_t i = 0; i <max_index; ++i)
    {
        M(i) *= scalar;
    }


}




#endif
