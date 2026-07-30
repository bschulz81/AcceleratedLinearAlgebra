#ifndef MATHFUNCTIONS_HPP
#define MATHFUNCTIONS_HPP

#include "mathfunctions.h"

#include "host_memory_functions.h"
#include "gpu_memory_functions.h"
#include "gpu_mathfunctions.h"
#include "inkernel_mathfunctions.h"


#include "mathfunctionspolicy.h"


template <typename T>
void Math_Functions::matrix_multiply_dot_accumulate( const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();

    if (policy.should_use_gpu(A, B, C, Math_Functions_Policy::default_cubic_treshold))
    {
        GPU_Math_Functions::matrix_multiply_dot_accumulate_g(A,B,C, policy.devicenum,policy.update_host);
    }
    else
        In_Kernel_Mathfunctions::matrix_multiply_dot_accumulate(A,B,C);
}



template <typename T>
void Math_Functions::matrix_multiply_dot( const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();

    if (policy.should_use_gpu(A, B, C, Math_Functions_Policy::default_cubic_treshold))
    {
        GPU_Math_Functions::matrix_multiply_dot_g(A,B,C, policy.devicenum,policy.update_host);
    }
    else
        In_Kernel_Mathfunctions::matrix_multiply_dot(A,B,C);
}


template <typename T>
void Math_Functions::matrix_add( const DataBlock<T>& A,const DataBlock<T>& B,  DataBlock<T>& C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(A, B, C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::matrix_add_g(A,B,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::matrix_add(A,B,C);
}

template <typename T>
void Math_Functions::matrix_add_accumulate( DataBlock<T>& A,const DataBlock<T>& B,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(A, B, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::matrix_add_accumulate_g(A,B, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::matrix_add_accumulate(A,B);
}


template <typename T>
void Math_Functions::matrix_subtract(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(A, B, C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::matrix_subtract_g(A,B,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::matrix_subtract(A,B,C);

}


template <typename T>
void Math_Functions::matrix_subtract_accumulate(  DataBlock<T>& A, const DataBlock<T>& B,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(A, B,  Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::matrix_subtract_accumulate_g(A,B, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::matrix_subtract_accumulate(A,B);

}


template <typename T>
void Math_Functions::matrix_multiply_vector( const DataBlock<T>&M, const DataBlock<T> V, DataBlock<T> C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(M, V, C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::matrix_multiply_vector_g(M,V,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::matrix_multiply_vector(M,V,C);
}

template <typename T>
void Math_Functions::matrix_multiply_vector(  const DataBlock<T>&M, const T*V, DataBlock<T> & C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(M,C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::matrix_multiply_vector_g(M,V,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::matrix_multiply_vector(M,V,C);
}




template <typename T>
void Math_Functions::matrix_multiply_scalar( const  DataBlock<T>& M, const T V, DataBlock<T>& C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(M,C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::matrix_multiply_scalar_g(M,V,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::matrix_multiply_scalar(M,V,C);
}




template <typename T>
void Math_Functions::matrix_multiply_scalar_accumulate( DataBlock<T>& M, const T V,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(M, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::matrix_multiply_scalar_accumulate_g(M,V, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::matrix_multiply_scalar_accumulate(M,V);
}

template <typename T>
void Math_Functions::vector_multiply_scalar( const  DataBlock<T>& M, const T V, DataBlock<T>& C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(M,C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::vector_multiply_scalar_g(M,V,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::vector_multiply_scalar(M,V,C);
}


template <typename T>
void Math_Functions::vector_multiply_scalar_accumulate(  DataBlock<T>& vec,const T scalar,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(vec, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::matrix_multiply_scalar_accumulate_g(vec,scalar, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::matrix_multiply_scalar_accumulate(vec,scalar);
}




template <typename T>
inline void Math_Functions::vector_add(  const DataBlock<T>& vec1, const DataBlock<T>& vec2, DataBlock<T> & res,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(vec1,vec2,res, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::vector_add_g(vec1,vec2,res, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::vector_add(vec1,vec2,res);
}


template <typename T>
inline void Math_Functions::vector_subtract(const DataBlock<T>& vec1,const DataBlock<T>& vec2, DataBlock<T> & res, const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(vec1,vec2,res, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::vector_subtract_g(vec1,vec2,res, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions::vector_subtract(vec1,vec2,res);
}


template <typename T>
inline T Math_Functions::dot_product( const DataBlock<T> &vec1, const DataBlock<T> &vec2, const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(vec1,vec2, Math_Functions_Policy::default_square_treshold))
        return GPU_Math_Functions::dot_product_g(vec1,vec2, policy.devicenum);
    else
        return In_Kernel_Mathfunctions::dot_product(vec1,vec2);
}


template <typename T>
void Math_Functions::cholesky_decomposition(const DataBlock<T> & A, DataBlock<T> & L, const Math_Functions_Policy*pol)
{

    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(A,L, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions::cholesky_decomposition_g(A,L, policy.devicenum,policy.update_host,policy.initialize_output_to_zeros);
    else
    {
        In_Kernel_Mathfunctions::cholesky_decomposition(A,L,policy.initialize_output_to_zeros);
    }

}

template <typename T>
void Math_Functions::lu_decomposition(const DataBlock<T>& A, DataBlock<T> &L,DataBlock<T>& U, const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy&policy =  (pol != nullptr) ? *pol : get_default_policy();

    if (policy.should_use_gpu(A,L,U, Math_Functions_Policy::default_cubic_treshold))
        GPU_Math_Functions::lu_decomposition_g(A,L,U, policy.devicenum,policy.update_host,policy.initialize_output_to_zeros);
    else
    {
        In_Kernel_Mathfunctions::lu_decomposition(A,L,U,policy.initialize_output_to_zeros);
    }

}
// Fast QR Decomposition Algorithm for mdspan
template <typename T>
void Math_Functions::qr_decomposition(const DataBlock<T>& A, DataBlock<T>& Q, DataBlock<T>& R,   const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy&policy =  (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(A,Q,R, Math_Functions_Policy::default_cubic_treshold))
        GPU_Math_Functions::qr_decomposition_g(A,Q,R, policy.devicenum,policy.update_host,policy.initialize_output_to_zeros,policy.memmapped_files);
    else
        In_Kernel_Mathfunctions::qr_decomposition(A,Q,R,policy.initialize_output_to_zeros,policy.memmapped_files);

}
#endif
