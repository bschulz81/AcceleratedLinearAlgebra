#ifndef MATHFUNCTIONS_HPP
#define MATHFUNCTIONS_HPP



#include "host_memory_functions.h"
#include "gpu_memory_functions.h"
#include "gpu_mathfunctions.h"

#include "inkernel_mathfunctions.h"

template<typename T>
inline void Math_Functions::matrix_multiply_dot(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
        const T CoefficientB,const T CoefficientC,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_matrix_multiply(A,B,C))
    {
        GPUOptions options
        {
            policy.devicenum,policy.update_host
        };

        if(policy.accumulation_precision > 1)
            GPU_Math_Functions::matrix_multiply_dot_kahan_g(A,B,C,CoefficientB,CoefficientC,options);
        else
            GPU_Math_Functions::matrix_multiply_dot_g(A,B,C,CoefficientB,CoefficientC,options);
    }
    else
    {
        if(policy.accumulation_precision > 1)
            In_Kernel_Mathfunctions::matrix_multiply_dot_kahan(A,B,C,CoefficientB,CoefficientC);
        else
            In_Kernel_Mathfunctions::matrix_multiply_dot(A,B,C,CoefficientB,CoefficientC);
    }
}



template<typename T>
inline void Math_Functions::matrix_multiply_vector(const DataBlock<T>& A,const DataBlock<T>& x,DataBlock<T>& y,
        const T Coefficientx,const T Coefficienty,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_matrix_vector(A,x,y))
    {
        GPUOptions options
        {
            policy.devicenum,policy.update_host
        };

        if(policy.accumulation_precision > 1)
            GPU_Math_Functions::matrix_multiply_vector_kahan_g(A,x,y,Coefficientx,Coefficienty,options);
        else
            GPU_Math_Functions::matrix_multiply_vector_g(A,x,y,Coefficientx,Coefficienty,options);
    }
    else
    {
        if(policy.accumulation_precision > 1)
            In_Kernel_Mathfunctions::matrix_multiply_vector_kahan(A,x,y,Coefficientx,Coefficienty);
        else
            In_Kernel_Mathfunctions::matrix_multiply_vector(A,x,y,Coefficientx,Coefficienty);
    }
}



template<typename T>
inline void Math_Functions::matrix_linear_combination(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
        const T CoefficientA,const T CoefficientB,const T CoefficientC,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_matrix(A,B,C))
    {
        GPU_Math_Functions::matrix_linear_combination_g(A,B,C,CoefficientA,CoefficientB,CoefficientC, GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::matrix_linear_combination(A,B,C,CoefficientA,CoefficientB,CoefficientC);
    }
}


template<typename T>
inline void Math_Functions::matrix_linear_combination(const DataBlock<T>& A,DataBlock<T>& C,const T CoefficientA,const T CoefficientC,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_matrix(A,C))
    {
        GPU_Math_Functions::matrix_linear_combination_g(A,C,CoefficientA,CoefficientC, GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::matrix_linear_combination(A,C,CoefficientA,CoefficientC);
    }
}

template<typename T>
inline void Math_Functions::matrix_multiply_scalar(const DataBlock<T>& M,const T scalar,DataBlock<T>& C,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_matrix(M,C))
    {
        GPU_Math_Functions::matrix_multiply_scalar_g(M,scalar,C,   GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::matrix_multiply_scalar(M,scalar,C);
    }
}

template<typename T>
inline void Math_Functions::matrix_multiply_scalar(DataBlock<T>& M,const T scalar,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_matrix(M))
{
    GPU_Math_Functions::matrix_multiply_scalar_g(M,scalar,GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::matrix_multiply_scalar(M,scalar);
    }
}



template<typename T>
inline void Math_Functions::vector_multiply_scalar(const DataBlock<T>& vec,const T scalar,DataBlock<T>& res,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_vector(vec,res))
    {
        GPU_Math_Functions::vector_multiply_scalar_g(vec,scalar,res,GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::vector_multiply_scalar(vec,scalar,res);
    }
}


template<typename T>
inline void Math_Functions::vector_linear_combination(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
        const T CoefficientA,const T CoefficientB,const T CoefficientC,const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_vector(A,B,C))
    {
        GPU_Math_Functions::vector_linear_combination_g(A,B,C,CoefficientA,CoefficientB,CoefficientC,
                GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::vector_linear_combination(A,B,C,CoefficientA,CoefficientB,CoefficientC);
    }
}


template<typename T>
inline void Math_Functions::vector_linear_combination(const DataBlock<T>& A,DataBlock<T>& C,const T CoefficientA,const T CoefficientC,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_vector(A,C))
    {
        GPU_Math_Functions::vector_linear_combination_g(A,C,CoefficientA,CoefficientC,
                GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::vector_linear_combination(A,C,CoefficientA,CoefficientC);
    }
}








template<typename T>
inline T Math_Functions::vector_dot_product(const DataBlock<T>& A,const DataBlock<T>& B,const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_vector(A,B))
    {
        GPUOptions options
        {
            policy.devicenum,
            policy.update_host
        };

        if(policy.accumulation_precision > 1)
            return GPU_Math_Functions::vector_dot_product_kahan_g(A,B,options);
        else
            return GPU_Math_Functions::vector_dot_product_g(A,B,options);
    }
    else
    {
        if(policy.accumulation_precision > 1)
            return In_Kernel_Mathfunctions::vector_dot_product_kahan(A,B);
        else
            return In_Kernel_Mathfunctions::vector_dot_product(A,B);
    }
}

template<typename T>
inline void Math_Functions::matrix_multiply_dot_sparse(
    const BlockedDataView<T>& Ablocks,const DataBlock<T>& B,DataBlock<T>& C,const T CoefficientB,const T CoefficientC,const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_sparse_matrix_multiply(Ablocks,B,C))
    {
        GPU_Math_Functions::matrix_multiply_dot_sparse_g(Ablocks,B,C,CoefficientB,CoefficientC,
                GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::matrix_multiply_dot_sparse(
            Ablocks,B,C,CoefficientB,CoefficientC);
    }
}


template<typename T>
inline void Math_Functions::matrix_multiply_vector_sparse(const BlockedDataView<T>& A,const DataBlock<T>& x,DataBlock<T>& y,
        const T Coefficientx,const T Coefficienty,const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_sparse_matrix_vector(A,x,y))
    {
        GPU_Math_Functions::matrix_multiply_vector_sparse_g(A,x,y,Coefficientx,Coefficienty,
                GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::matrix_multiply_vector_sparse(A,x,y,Coefficientx,Coefficienty);
    }
}


template<typename T>
inline void Math_Functions::matrix_multiply_vector_sparse(const BlockedDataView<T>& A,const BlockedDataView<T>& x,DataBlock<T>& y,
        const T Coefficientx,const T Coefficienty,const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_sparse_matrix_vector(A,x,y))
    {
        GPU_Math_Functions::matrix_multiply_vector_sparse_g(A,x,y,Coefficientx,Coefficienty,
                GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::matrix_multiply_vector_sparse(A,x,y,Coefficientx,Coefficienty);
    }
}



template<typename T>
inline void Math_Functions::vector_multiply_scalar(DataBlock<T>& vec,const T scalar,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_vector(vec,vec,vec))
    {
        GPU_Math_Functions::vector_multiply_scalar_g(vec,scalar,
                GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::vector_multiply_scalar(vec,scalar);
    }
}

template<typename T>
inline void Math_Functions::cholesky_decomposition(const DataBlock<T>& A,DataBlock<T>& L,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    const bool initialize_to_zero =
        policy.initialize_output_to_zeros;

    if(policy.should_use_gpu_decomposition(A))
    {
        GPU_Math_Functions::cholesky_decomposition_g(A,L,initialize_to_zero,
                GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::cholesky_decomposition(A,L,initialize_to_zero);
    }
}


template<typename T>
inline void Math_Functions::lu_decomposition(const DataBlock<T>& A,DataBlock<T>& L,DataBlock<T>& U,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    const bool initialize_to_zero =
        policy.initialize_output_to_zeros;


    if(policy.should_use_gpu_decomposition(A))
    {
        GPU_Math_Functions::lu_decomposition_g(A,L,U,initialize_to_zero,
                                               GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::lu_decomposition(A,L,U,initialize_to_zero);
    }
}

template<typename T>
inline void Math_Functions::qr_decomposition(const DataBlock<T>& A,DataBlock<T>& Q,DataBlock<T>& R,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();


    const bool initialize_to_zero =
        policy.initialize_output_to_zeros;


    const bool with_memmaps =
        policy.memmapped_files;

    if(policy.should_use_gpu_decomposition(A))
    {
        GPU_Math_Functions::qr_decomposition_g(A,Q,R,initialize_to_zero,with_memmaps,GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::qr_decomposition(A,Q,R,initialize_to_zero,with_memmaps);
    }
}

template<typename T>
inline void Math_Functions::matrix_multiply_dot_sparse(const BlockedDataView<T>& A,const BlockedDataView<T>& B,DataBlock<T>& C,
        const T CoefficientB,const T CoefficientC,const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();


    if(policy.should_use_gpu_sparse_matrix_multiply(A,B,C))
    {
        GPU_Math_Functions::matrix_multiply_dot_sparse_g(A,B,C,CoefficientB,CoefficientC,
                GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::matrix_multiply_dot_sparse(A,B,C,CoefficientB,CoefficientC);
    }
}


template<typename T>
inline void Math_Functions::tensor_linear_combination(const DataBlock<T>& A,const DataBlock<T>& B,DataBlock<T>& C,
        const T CoefficientA,const T CoefficientB,const T CoefficientC,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_elementwise(A,B,C))
    {
        GPU_Math_Functions::tensor_linear_combination_g(A,B,C,CoefficientA,CoefficientB,CoefficientC, GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::tensor_linear_combination(A,B,C,CoefficientA,CoefficientB,CoefficientC);
    }
}


template<typename T>
inline void Math_Functions::tensor_linear_combination(const DataBlock<T>& A,DataBlock<T>& C,const T CoefficientA,const T CoefficientC,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_elementwise(A,C))
    {
        GPU_Math_Functions::tensor_linear_combination_g(A,C,CoefficientA,CoefficientC, GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::tensor_linear_combination(A,C,CoefficientA,CoefficientC);
    }
}

template<typename T>
inline void Math_Functions::tensor_multiply_scalar(const DataBlock<T>& M,const T scalar,DataBlock<T>& C,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_elementwise(M,C))
    {
        GPU_Math_Functions::tensor_multiply_scalar_g(M,scalar,C,   GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::tensor_multiply_scalar(M,scalar,C);
    }
}

template<typename T>
inline void Math_Functions::tensor_multiply_scalar(DataBlock<T>& M,const T scalar,
        const Math_Functions_Policy* pol)
{
    const auto& policy = pol ? *pol : get_default_policy();

    if(policy.should_use_gpu_elementwise(M))
    {
        GPU_Math_Functions::tensor_multiply_scalar_g(M,scalar,GPUOptions{.device=policy.devicenum,.update_host=policy.update_host});
    }
    else
    {
        In_Kernel_Mathfunctions::tensor_multiply_scalar(M,scalar);
    }
}


#endif
