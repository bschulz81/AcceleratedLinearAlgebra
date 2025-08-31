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
    inline static void matrix_add_g( datastruct<T>& A, datastruct<T>& B, datastruct<T>& C,int dev,bool update_host=true);
    inline static void matrix_subtract_g( datastruct<T>& A,  datastruct<T>& B, datastruct<T>& C,int dev,bool update_host=true);

    inline static void matrix_multiply_vector_g(  datastruct<T>&M,  datastruct<T> V, datastruct<T> C,int dev,bool update_host=true);
    inline static void matrix_multiply_vector_g(  datastruct<T>M, T*V, datastruct<T> & C, int dev,bool update_host=true);
    inline static void matrix_multiply_scalar_g(   datastruct<T>& M, T V, datastruct<T>& C, int dev,bool update_host=true);

    inline static void vector_multiply_scalar_g( datastruct<T>& vec,T scalar,datastruct<T>& res,int dev,bool update_host=true);
    inline static void vector_add_g(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res,int dev,bool update_host=true);
    inline static void vector_subtract_g(  datastruct<T>& vec1,  datastruct<T>& vec2, datastruct<T> & res,  int dev,bool update_host=true);

    inline static T dot_product_g(  datastruct<T> &vec1,  datastruct<T> &vec2, int dev);

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

    #pragma omp target teams distribute parallel for collapse(2) shared(A,B,C,rows,cols,inner_dim) device(dev)
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

template <typename T>
void GPU_Math_Functions<T>::matrix_add_g( datastruct<T>& A, datastruct<T>& B, datastruct<T>& C,int dev,bool update_host)
{

    const size_t n=A.dpextents[0];
    const size_t m=A.dpextents[1];

    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperA(A,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperB(B,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperC(C,dev,true,update_host);

    #pragma omp target teams distribute parallel for shared(C,A,B,n,m)device(dev)
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

    #pragma omp target teams distribute parallel for shared(A,B,C,n,m)device(dev)
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

    #pragma omp target teams distribute parallel for shared(C,M,V,m,n)device(dev)
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

    #pragma omp target teams distribute parallel for shared(C,M,V,m,n)device(dev)
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

    #pragma omp target teams distribute parallel for shared(C,M,V,n,m) device(dev)
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

    #pragma omp target teams distribute parallel for simd shared(res,vec,n,scalar)device(dev)
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

    #pragma omp target teams distribute parallel for simd shared(res,vec1,vec2,n)device(dev)
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

    #pragma omp target teams distribute parallel for simd device(dev)
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

    #pragma omp target teams distribute parallel for simd shared(vec1,vec2,n) reduction(+:result)device(dev)
    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }


    return result;
}


template <typename T>
void GPU_Math_Functions<T>::cholesky_decomposition_g(datastruct<T> & A,datastruct<T> & L,int dev,bool update_host, bool initialize_output_to_zero)
{


    const size_t n = A.dpextents[0];



    //these functions check isdevptr to see whether data was allocated with malloc. they do only offload if that is not the case.
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperA(A,dev,false,false);
    typename Datastruct_GPU_Memory_Functions<T>::OffloadHelper offloadhelperL(L,dev,true,update_host);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) shared(L,n) device(dev)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=0;
            }
    }

    for (size_t c = 0; c < n; ++c)
    {
        T tmp=0,temp4=0;
        #pragma omp target teams distribute  parallel for simd reduction(+:tmp) shared(L,c)map(tofrom:tmp)device(dev)
        for (size_t k = 0; k < c; ++k)
        {
            const T tmp3=L(c,k);
            tmp+= tmp3 * tmp3;
        }


//        size_t offset_A =c * A.dpstrides[0]+c*A.dpstrides[1]; // host-side
//        size_t offset_L =c * L.dpstrides[0]+c*L.dpstrides[1];
//        T* Adevptr=(T*)omp_get_mapped_ptr(A.dpdata,dev);
//        T* Ldevptr=(T*)omp_get_mapped_ptr(L.dpdata,dev);
//        T Acc=0;
//        omp_target_memcpy(&Acc, Adevptr, sizeof(T), 0, offset_A,omp_get_initial_device(),dev);
//
//        temp4=Acc-tmp;
//        temp4=sqrt(temp4);
//
//        omp_target_memcpy(Ldevptr, &temp4, sizeof(T), offset_L,0,dev, omp_get_initial_device());

        #pragma omp target map(to:tmp)map(from:temp4) device(dev)
        {
            temp4=A(c, c)-tmp;
            temp4=sqrt(temp4);
            L(c,c)=temp4;
        }

        #pragma omp target teams distribute parallel for shared(A,L,c) map(to:temp4) device(dev)
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
        #pragma omp target teams distribute parallel for simd collapse(2) shared(L,U,n) device(dev)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=0;
                U(i,j)=0;
            }
    }

    size_t z=0;
    for (size_t c = 0; c < n; ++c)
    {
        #pragma omp target teams distribute shared(A,c,L,U)device(dev)
        for (size_t i = c; i < n; ++i)
        {
            T temp=A(c,i);
            #pragma omp parallel for simd reduction(-:temp) shared(c,L,U)
            for (size_t k = z; k < c; ++k)
            {
                temp -= U( k,i) * L( c,k);
            }
            U(c,i)=temp;
        }

        #pragma omp target teams distribute shared(A,c,L,U)device(dev)
        for (size_t i = c; i < n; ++i)
        {
            const T temp4=U(c,c);
            T temp = A(i,c);
            #pragma omp parallel for simd reduction (-:temp)shared(U,L,c)
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
    datastruct<T> M=Datastruct_GPU_Memory_Functions<T>::alloc_data_copy_strides_extents_device(A.dpdatalength,A.dprowmajor, A.dprank,A.dpextents,A.dpstrides,
                    memmap_tempfiles,dev);
    Datastruct_GPU_Memory_Functions<T>::create_in_struct(M,dev);

    if(initialize_output_to_zero)
    {
        #pragma omp target teams distribute parallel for shared(Q,M,A,R,n,m)device(dev)
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
        #pragma omp target teams distribute parallel for shared(n,m,A,M)device(dev)
        for (size_t i = 0; i < n; ++i)
        {
            #pragma omp simd
            for (size_t j = 0; j < m; ++j)
            {
                M(i,j)=A(i,j);
            }
        }
    }


    size_t z = 0;


    for (size_t c = 0; c < m; ++c)
    {

        size_t pext0=M.dpextents[0];


        for (size_t j = z; j < c; ++j)
        {

            T dot_pr=0;

            #pragma omp target teams distribute parallel for simd shared(Q,M,pext0) reduction(+:dot_pr)device(dev)
            for (size_t i = 0; i < pext0; ++i)
            {
                size_t pextv[1];
                size_t pstrv[1];
                size_t pextu[1];
                size_t pstru[1];
                datastruct<T> u = Q.column(j,pextu,pstru);
                datastruct<T> v = M.column(c,pextv,pstrv);
                dot_pr += u(i) * v(i);
            }

            const T cdot_pr=dot_pr;
            #pragma omp target teams distribute  parallel for simd shared(Q,M, pext0,cdot_pr)device(dev)
            for (size_t i = 0; i < pext0; ++i)
            {
                size_t pextv[1];
                size_t pstrv[1];
                size_t pextu[1];
                size_t pstru[1];
                datastruct<T> u = Q.column(j,pextu,pstru);
                datastruct<T>  v = M.column(c,pextv,pstrv);
                v(i) -= cdot_pr * u(i);
            }
        }
        // Normalize v
        T norm=0;
        #pragma omp target teams distribute  parallel for simd shared(M,pext0)reduction(+: norm)device(dev)
        for (size_t i = 0; i < pext0; ++i)
        {
            size_t pextv[1];
            size_t pstrv[1];
            datastruct<T>  v = M.column(c,pextv,pstrv);
            norm += v(i) * v(i);
        }

        const T normc= sqrt(norm);
        #pragma omp target teams distribute  parallel for simd shared(M,pext0,normc) device(dev)
        for (size_t i = 0; i < pext0; ++i)
        {
            size_t pextv[1];
            size_t pstrv[1];
            datastruct<T>  v = M.column(c,pextv,pstrv);
            v(i)= v(i)/normc;
        }

        #pragma omp target teams distribute  parallel for simd shared(M,pext0) device(dev)
        for (size_t i = 0; i < pext0; ++i)
        {
            size_t pextv[1];
            size_t pstrv[1];
            datastruct<T>  v = M.column(c,pextv,pstrv);
            Q(i,c) = v(i);
        }
    }

    const size_t rows = Q.dpextents[0]; // Number of rows in A and C
    const size_t cols = A.dpextents[1]; // Number of columns in B and C
    const  size_t inner_dim = Q.dpextents[1]; // Number of columns in A and rows in B

    #pragma omp target teams distribute collapse(2) shared(Q,A,R,rows,cols, inner_dim)device(dev)
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
    Datastruct_GPU_Memory_Functions<T>::exit_struct(M,dev);
    Datastruct_GPU_Memory_Functions<T>::free_copy_device(M,memmap_tempfiles,dev);
}




#endif

