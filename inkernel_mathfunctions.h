#ifndef INKERNELMATHFUNCTIONS
#define INKERNELMATHFUNCTIONS

#include "cmath"
#include "datastruct.h"


using namespace std;



#pragma omp begin declare target
template <typename T>
class In_Kernel_Mathfunctions
{
public:
    inline static void cholesky_decomposition_w(const datastruct<T>& A, datastruct<T>& L,bool initialize_to_zero=true);
    inline static void lu_decomposition_w(const  datastruct<T>& dA, datastruct<T>& dL, datastruct<T>& dU,bool initialize_to_zero=true);
    inline static void qr_decomposition_w( const datastruct<T>&A, datastruct<T> Q, datastruct<T> &R,bool initialize_to_zero=true,bool with_memmaps=false);

    inline static void cross_product( const datastruct<T>& vec1,const   datastruct<T>& vec2, datastruct<T>& res);

    inline static void matrix_multiply_dot_w( const datastruct<T>& A,  const datastruct<T>& B, datastruct<T>& C);
    inline static void matrix_multiply_dot_w_kahan( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C);
    inline static void matrix_multiply_dot_v( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C);
    inline static void matrix_multiply_dot_s(const  datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C);

    inline static void matrix_add_w(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C);
    inline static void matrix_add_v(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C);
    inline static void matrix_add_s(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C);

    inline static void matrix_subtract_w(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C);
    inline static void matrix_subtract_v(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C);
    inline static void matrix_subtract_s(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C);

    inline static void matrix_multiply_vector_w( const datastruct<T>&M, const datastruct<T>& V, datastruct<T> C);
    inline static void matrix_multiply_vector_v( const datastruct<T>&M, const datastruct<T>& V, datastruct<T> C);
    inline static void matrix_multiply_vector_s( const datastruct<T>&M, const datastruct<T>& V, datastruct<T> C);

    inline static void matrix_multiply_vector_s( const datastruct<T>&M,const  T*V, datastruct<T> & C);
    inline static void matrix_multiply_vector_v( const datastruct<T>&M,const  T*V, datastruct<T> & C);
    inline static void matrix_multiply_vector_w( const datastruct<T>&M,const  T*V, datastruct<T> & C);

    inline static void vector_add_s(const  datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res);
    inline static void vector_add_v( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res);
    inline static void vector_add_w( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res);

    inline static void vector_subtract_w(const  datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res);
    inline static void vector_subtract_v( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res);
    inline static void vector_subtract_s( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res);

    inline static T dot_product_s( const datastruct<T> &vec1,const  datastruct<T> &vec2);
    inline static T dot_product_v( const datastruct<T> &vec1, const datastruct<T> &vec2);
    inline static T dot_product_w( const datastruct<T> &vec1,const  datastruct<T> &vec2);
    inline static T dot_product_w_kahan(const  datastruct<T> &vec1, const datastruct<T> &vec2);

    inline static void matrix_multiply_scalar_s( const  datastruct<T>& M, const T V, datastruct<T>& C);
    inline static void matrix_multiply_scalar_v( const  datastruct<T>& M,const  T V, datastruct<T>& C);
    inline static void matrix_multiply_scalar_w( const  datastruct<T>& M,const  T V, datastruct<T>& C);

    inline static void vector_multiply_scalar_s( const datastruct<T>& vec,const T scalar,datastruct<T>& res);
    inline static void vector_multiply_scalar_v( const datastruct<T>& vec,const T scalar,datastruct<T>& res);
    inline static void vector_multiply_scalar_w( const datastruct<T>& vec,const T scalar,datastruct<T>& res);

    inline static T  kahan_sum(const T *arr,size_t n);
    inline static T  neumaier_sum(const T*arr,size_t n);
};
#pragma omp end declare target

#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::cholesky_decomposition_w(const datastruct<T>& A, datastruct<T>& L, bool initialize_to_zero)
{

    const size_t n = A.dpextents[0];

    if(initialize_to_zero)
    {
        #pragma omp parallel for simd collapse(2) shared(L)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=0;
            }
    }

    for (size_t c = 0; c < n; ++c)
    {
        T tmp=0;

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
void In_Kernel_Mathfunctions<T>::lu_decomposition_w(const  datastruct<T>& A, datastruct<T>& L, datastruct<T>& U,bool initialize_to_zero)
{

    const size_t n = A.dpextents[0];


    if(initialize_to_zero)
    {
        #pragma omp parallel for simd collapse(2) shared(L,U)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j <n; ++j)
            {
                L(i,j)=0;
                U(i,j)=0;
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
void In_Kernel_Mathfunctions<T>::qr_decomposition_w( const datastruct<T>&A, datastruct<T> Q, datastruct<T> &R,bool initialize_to_zero, bool with_memmaps)
{
    const size_t n = A.dpextents[0];
    const size_t m = A.dpextents[1];


    T* __restrict tempM;

    if(with_memmaps)
        tempM=Datastruct_Host_Memory_Functions<T>::create_temp_mmap(A.dpdatalength);
    else
        tempM=(T*)malloc(sizeof(T)*A.dpdatalength);

    size_t Mext[2]= {A.dpextents[0],A.dpextents[1]};
    size_t Mstrides[2]= {A.dpstrides[0],A.dpstrides[1]};

    datastruct<T> M(tempM,A.dpdatalength,A.dprowmajor,A.dprank,Mext,Mstrides,false,false,false);


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
        datastruct<T> v = M.column_rr(c,pextv,pstrv);
        for (size_t j = 0; j < c; ++j)
        {
            size_t pextu[1];
            size_t pstru[1];

            T dot_pr=0;
            datastruct<T> u = Q.column_rr(j,pextu,pstru);
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
        T norm=0;
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
        Datastruct_Host_Memory_Functions<T>::delete_temp_mmap(tempM,A.dpdatalength);
    else
        free(tempM);
}
#pragma omp end declare target




#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::cross_product( const datastruct<T>& vec1, const  datastruct<T>& vec2, datastruct<T>& res)
{
    res(0) = vec1(1) * vec2(2) - vec1(2) * vec2(1);
    res(1) = vec1(2) * vec2(0) - vec1(0) * vec2(2);
    res(2) = vec1(0) * vec2(1) - vec1(1) * vec2(0);

}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w_kahan( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C)
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
}
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_v( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_multiply_dot_s( const datastruct<T>& A, const  datastruct<T>& B, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_add_w(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_add_v(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_add_s(const datastruct<T>& A,const datastruct<T>& B, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_subtract_w(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_subtract_v(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_subtract_s(const datastruct<T>& A,const  datastruct<T>& B, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_w( const datastruct<T>&M,const  datastruct<T>& V, datastruct<T> C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    #pragma omp parallel for shared(M,V,C)
    for (size_t i = 0; i <n; ++i)
    {
        T sum=0;
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
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_v( const datastruct<T>&M,const  datastruct<T>& V, datastruct<T> C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        T sum=0;
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
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_s( const datastruct<T>&M,const  datastruct<T>& V, datastruct<T> C)
{

    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        T sum=0;
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
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_s( const datastruct<T>&M,const  T*V, datastruct<T> & C)
{
    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        T sum=0;
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
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_v( const datastruct<T>&M, const T*V, datastruct<T> & C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];

    for (size_t i = 0; i <n; ++i)
    {
        T sum=0;
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
void In_Kernel_Mathfunctions<T>::matrix_multiply_vector_w( const datastruct<T>&M, const T*V, datastruct<T> & C)
{


    const size_t n= M.dpextents[0];
    const size_t m=M.dpextents[1];
    #pragma omp parallel for shared(M,V,C)
    for (size_t i = 0; i <n; ++i)
    {
        T sum=0;
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
void In_Kernel_Mathfunctions<T>::vector_add_s( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
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
void In_Kernel_Mathfunctions<T>::vector_add_v( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
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
void In_Kernel_Mathfunctions<T>::vector_add_w( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
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
void In_Kernel_Mathfunctions<T>::vector_subtract_w( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
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
void In_Kernel_Mathfunctions<T>::vector_subtract_v( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
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
void In_Kernel_Mathfunctions<T>::vector_subtract_s( const datastruct<T>& vec1,const  datastruct<T>& vec2, datastruct<T> & res)
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
T In_Kernel_Mathfunctions<T>::dot_product_s(const  datastruct<T> &vec1, const datastruct<T> &vec2)
{
    const size_t n=vec1.dpextents[0];
    T result=0;

    for (size_t i = 0; i < n; ++i)
    {
        result += vec1(i) * vec2(i);
    }
    return result;
}
#pragma omp end declare target


#pragma omp begin declare target
template <typename T>
T In_Kernel_Mathfunctions<T>::dot_product_v(const  datastruct<T> &vec1, const datastruct<T> &vec2)
{
    const size_t n=vec1.dpextents[0];

    T result=0;
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
T In_Kernel_Mathfunctions<T>::dot_product_w(const  datastruct<T> &vec1, const datastruct<T> &vec2)
{
    const size_t n=vec1.dpextents[0];
    T result=0;
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
T In_Kernel_Mathfunctions<T>::dot_product_w_kahan(const  datastruct<T> &vec1, const datastruct<T> &vec2)
{

    const size_t n=vec1.dpextents[0];

    int total_threads =   omp_get_max_threads();

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

        #pragma omp parallel for
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
#pragma omp end declare target



#pragma omp begin declare target
template <typename T>
void In_Kernel_Mathfunctions<T>::matrix_multiply_scalar_s(  const datastruct<T>& M, const T V, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_multiply_scalar_v(  const datastruct<T>& M, const T V, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::matrix_multiply_scalar_w(  const datastruct<T>& M, const T V, datastruct<T>& C)
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
void In_Kernel_Mathfunctions<T>::vector_multiply_scalar_s( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
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
void In_Kernel_Mathfunctions<T>::vector_multiply_scalar_v( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
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
void In_Kernel_Mathfunctions<T>::vector_multiply_scalar_w( const datastruct<T>& vec,const T scalar,datastruct<T>& res)
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
    double sum = 0.0;
    double c = 0.0; // compensation

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
    double sum = 0.0;
    double comp = 0.0;

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
