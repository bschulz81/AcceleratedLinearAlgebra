#ifndef GPUMATHFUNCTIONS
#define GPUMATHFUNCTIONS

#include "datastruct.h"
#include "datastruct_host_memory_functions.h"
#include "datastruct_gpu_memory_functions.h"

template <typename T>
class GPU_Math_Functions
{
public:
    inline static void matrix_multiply_dot_g(  datastruct<T>& A,  datastruct<T>& B,  datastruct<T>& C,int dev,bool update_host=true);
    inline static void matrix_multiply_dot_g_kahan(  datastruct<T>& A,  datastruct<T>& B,  datastruct<T>& C,int dev,bool update_host=true);
    inline static void matrix_add_g( datastruct<T>& A, datastruct<T>& B, datastruct<T>& C,int dev,bool update_host=true);
    inline static void matrix_subtract_g( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C,int dev,bool update_host=true);

    inline static void matrix_multiply_vector_g(  datastruct<T>&M,  datastruct<T> V, datastruct<T> C,int dev,bool update_host=true);
    inline static void matrix_multiply_vector_g(  datastruct<T>M, T*V, datastruct<T> & C, int dev,bool update_host=true);
    inline static void matrix_multiply_scalar_g(   datastruct<T>& M, T V, datastruct<T>& C, int dev,bool update_host=true);

    inline static void vector_multiply_scalar_g( datastruct<T>& vec,T scalar,datastruct<T>& res,int dev,bool update_host=true);
    inline static void vector_add_g(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res,int dev,bool update_host=true);
    inline static void vector_subtract_g(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res,  int dev,bool update_host=true);

    inline static T dot_product_g(  datastruct<T> &vec1,  datastruct<T> &vec2, int dev);
    inline static T dot_product_g_kahan(  datastruct<T> &vec1,  datastruct<T> &vec2,int dev, int nteams, int nthreads_per_team );

    inline static void cholesky_decomposition_g(datastruct<T>& A, datastruct<T> & L, int dev,bool update_host=true, bool initialize_output_to_zero=true);
    inline static void lu_decomposition_g(datastruct<T> &A, datastruct<T> & L,datastruct<T> & U, int dev,bool update_host=true,bool initialize_output_to_zero=true);
    inline static void qr_decomposition_g(datastruct<T> &A,datastruct<T>& Q, datastruct<T> & R,  int dev,bool update_host=true,bool initialize_output_to_zero=true,bool memmaptempfiles=false);

};


template <typename T>
void GPU_Math_Functions<T>::matrix_multiply_dot_g(  datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C,int dev,bool update_host)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadA(A, dev, false, false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadB(B, dev, false, false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadC(C, dev, true, update_host);

   const size_t Astr0=A.dpstrides[0];
   const size_t Astr1=A.dpstrides[1];
   const size_t Bstr0=B.dpstrides[0];
   const size_t Bstr1=B.dpstrides[1];
   const size_t Cstr0=C.dpstrides[0];
   const size_t Cstr1=C.dpstrides[1];
    #pragma omp target teams distribute parallel for collapse(2) shared(A,B,C) device(dev)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < inner_dim; ++k)
            {
                sum += A.dpdata[i*Astr0+k*Astr1] *B.dpdata[k*Bstr0+j*Bstr1];
            }
            C.dpdata[i*Cstr0+j*Cstr1]= sum;
        }


}



template <typename T>
void GPU_Math_Functions<T>::matrix_multiply_dot_g_kahan(  datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C,int dev,bool update_host)
{
    const size_t rows=A.dpextents[0];
    const size_t cols=B.dpextents[1];
    const size_t inner_dim=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadA(A, dev, false, false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadB(B, dev, false, false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadC(C, dev, true, update_host);

    #pragma omp target teams distribute parallel for simd collapse(2) shared(A,B,C) device(dev)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
        {
            T sum = 0;
            T c=0;
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
void GPU_Math_Functions<T>::matrix_add_g( datastruct<T>& A, datastruct<T>& B, datastruct<T>& C,int dev,bool update_host)
{

    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperA(A,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperB(B,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

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
void GPU_Math_Functions<T>::matrix_subtract_g( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C,int dev,bool update_host)
{
    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperA(A,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperB(B,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

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
void GPU_Math_Functions<T>::matrix_multiply_vector_g(  datastruct<T>&M,  datastruct<T> V, datastruct<T> C,int dev,bool update_host)
{


    const size_t n= M.dpextents[0];
    const size_t m=V.dpextents[0];



    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperM(M,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperV(V,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for shared(M,V,C) device(dev)
    for (size_t i = 0; i <n; ++i)
    {
        T sum=0;
        #pragma omp simd reduction(+: sum)
        for (size_t j = 0; j <m ; ++j)
        {
            sum= M(i, j) * V(j);
        }
        C(i)=sum;
    }


}

template <typename T>
void GPU_Math_Functions<T>::matrix_multiply_vector_g( const datastruct<T>M, T*V, datastruct<T> & C,int dev,bool update_host)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    #pragma omp target enter data map (to:V[0:n])device(dev)
    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperM(M,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for shared(M,V,C) device(dev)
    for (size_t i = 0; i <n; ++i)
    {
        T sum=0;
        #pragma omp simd reduction(+: sum)
        for (size_t j = 0; j <m ; ++j)
        {
            sum= M(i, j) * V(j);
        }
        C(i)=sum;
    }

    #pragma omp target exit data map (release:V[0:n])device(dev)

}




template <typename T>
void GPU_Math_Functions<T>::matrix_multiply_scalar_g(   datastruct<T>& M, T V, datastruct<T>& C,int dev,bool update_host)
{

    const size_t n=C.dpextents[0];
    const size_t m= C.dpextents[1];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperM(M,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

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
void GPU_Math_Functions<T>::vector_multiply_scalar_g( datastruct<T>& vec,T scalar,datastruct<T>& res,int dev,bool update_host)
{
    const size_t n=vec.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelpervec(vec,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperres(res,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd shared(res,vec) device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec(i)*scalar;
    }


}




template <typename T>
inline void GPU_Math_Functions<T>::vector_add_g(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res,int dev,bool update_host)
{
    const size_t n=vec1.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelpervec1(vec1,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelpervec2(vec2,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperres(res,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd shared(res,vec1,vec2) device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)+vec2(i);
    }

}


template <typename T>
inline void GPU_Math_Functions<T>::vector_subtract_g( datastruct<T>& vec1, datastruct<T>& vec2, datastruct<T> & res,int dev,bool update_host)
{
    const size_t n=vec1.dpextents[0];


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelpervec1(vec1,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelpervec2(vec2,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperres(res,dev,true,update_host);

    #pragma omp target teams distribute parallel for simd shared(vec1,vec2,res) device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        res(i) = vec1(i)-vec2(i);
    }



}


template <typename T>
inline T GPU_Math_Functions<T>::dot_product_g(  datastruct<T> &vec1,  datastruct<T> &vec2,int dev)
{
    const size_t n=vec1.dpextents[0];

    T result=0;



    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelpervec1(vec1,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelpervec2(vec2,dev,false,false);

    #pragma omp target teams distribute parallel for simd reduction(+:result)shared(vec1,vec2) device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }

    return result;
}


template <typename T>
inline T GPU_Math_Functions<T>::dot_product_g_kahan(  datastruct<T> &vec1,  datastruct<T> &vec2,int dev, int nteams, int nthreads_per_team)
{
    const size_t n=vec1.dpextents[0];

    int total_threads = nteams * nthreads_per_team;

    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelpervec1(vec1,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelpervec2(vec2,dev,false,false);

    if(n < (size_t)total_threads)
    {
        T result = 0.0;
        T c_final = 0.0;
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
            thread_sums[idx] = 0.0;
            thread_cs[idx] = 0.0;
        }


        #pragma omp target teams distribute parallel for map(tofrom: thread_sums[0:total_threads], thread_cs[0:total_threads]) device(dev)
        for (int tid = 0; tid < total_threads; ++tid)
        {
            T local_sum = 0.0;
            T c = 0.0;

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

        T result = 0.0;
        T c_final = 0.0;

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
void GPU_Math_Functions<T>::cholesky_decomposition_g(datastruct<T> & A,datastruct<T> & L,int dev,bool update_host, bool initialize_output_to_zero)
{


    const size_t n = A.dpextents[0];



    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperA(A,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperL(L,dev,true,update_host);

    T* dataA=(T*)omp_get_mapped_ptr(A.dpdata,dev);
    T* dataL=(T*)omp_get_mapped_ptr(L.dpdata,dev);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd collapse(2)  shared(L) device(dev)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=0;
            }
    }

    for (size_t c = 0; c < n; ++c)
    {

        T tmp=0;

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
void GPU_Math_Functions<T>::lu_decomposition_g(datastruct<T>& A, datastruct<T> &L,datastruct<T>& U,int dev, bool update_host,bool initialize_output_to_zero)
{


    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperA(A,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperL(L,dev,true,update_host);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperU(U,dev,true,update_host);

    size_t n = A.dpextents[0];



    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) shared(U,L) device(dev)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=0;
                U(i,j)=0;
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
        T temp4=0;
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
void GPU_Math_Functions<T>::qr_decomposition_g(datastruct<T>& A, datastruct<T>& Q, datastruct<T>& R,  int dev,bool update_host,bool initialize_output_to_zero, bool memmap_tempfiles)
{
    const size_t n = A.dpextents[0];
    const size_t m = A.dpextents[1];

    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperA(A,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperQ(Q,dev,true,update_host);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperR(R,dev,true,update_host);


    datastruct<T> tQ=Q;

    if(!tQ.dpdata_is_devptr)
        tQ.dpdata=(T*) omp_get_mapped_ptr(Q.dpdata,dev);
    tQ.dpdata_is_devptr=true;


    datastruct<T> M=Datastruct_GPU_Memory_Functions<T>::alloc_data_copy_strides_extents_device(A.dpdatalength,A.dprowmajor, A.dprank,A.dpextents,A.dpstrides,
                    memmap_tempfiles,dev);

    Datastruct_GPU_Memory_Functions<T>::create_in_struct(M,dev);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for shared(tQ,M,A,R) device(dev)
        for (size_t i = 0; i < n; ++i)
        {
            #pragma omp simd
            for (size_t j = 0; j < n; ++j)
                tQ(i,j) = 0;
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
        #pragma omp target teams distribute parallel for shared(M,A) device(dev)
        for (size_t i = 0; i < n; ++i)
        {
            #pragma omp simd
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

        datastruct<T> v = M.column_rr(c, pextv, pstrv);  // current column, updated in place
        typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperv(v,dev,false,false);


        for (size_t j = 0; j < c; ++j)
        {
            size_t pextu[1], pstru[1];

            datastruct<T> u = tQ.column_rr(j, pextu, pstru);
            typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperu(u,dev,false,false);
            T dot_pr = 0;

            #pragma omp target teams distribute parallel for simd reduction(+:dot_pr)shared(u,v) device(dev)
            for (size_t i = 0; i < pext0; ++i)
                dot_pr += u(i) * v(i);

            const T cdot_pr = dot_pr;
            #pragma omp target teams distribute parallel for simd shared(u,v) device(dev)
            for (size_t i = 0; i < pext0; ++i)
                v(i) -= cdot_pr * u(i);
        }

        T norm = 0;
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

    #pragma omp target teams distribute parallel for collapse(2) shared(tQ,A,R) device(dev)
    for (size_t i = 0; i < rows; ++i)
    {
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
    Datastruct_GPU_Memory_Functions<T>::exit_struct(M,dev);
    Datastruct_GPU_Memory_Functions<T>::free_copy_device(M,memmap_tempfiles,dev);
}




#endif

