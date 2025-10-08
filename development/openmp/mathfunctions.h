#ifndef MATHFUNCTIONS
#define MATHFUNCTIONS

#include "datablock.h"
#include "mdspan_omp.h"
#include "mdspan_data.h"

#include "datablock_host_memory_functions.h"
#include "datablock_gpu_memory_functions.h"

#include "gpu_mathfunctions.h"
#include "inkernel_mathfunctions.h"

struct DeviceInfo {
    int dev_id;
    int num_teams;
    int threads_per_team;
};

// Query function
inline void query_device_team_thread_counts(int dev, DeviceInfo &info) {
    info.dev_id = dev;
    info.num_teams = 0;
    info.threads_per_team = 0;

    #pragma omp target map(from: info) device(dev)
    {
        #pragma omp teams
        {
            if (omp_get_team_num() == 0) {
                info.num_teams = omp_get_num_teams();
            }
            #pragma omp parallel
            {
                if (omp_get_thread_num() == 0) {
                    info.threads_per_team = omp_get_num_threads();
                }
            }
        }
    }
}






class Math_Functions_Policy
{
    public:
    enum Mode { CPU_ONLY, GPU_ONLY, AUTO } mode = AUTO;
    bool update_host = true;
    bool memmapped_files = false;
    bool initialize_output_to_zeros = true;
    size_t precision=1;
    int devicenum = omp_get_default_device();
    int num_gpus = 0;

    static constexpr size_t max_problem_size_for_gpu = SIZE_MAX;
    static constexpr size_t default_cubic_treshold = 256;
    static constexpr size_t default_square_treshold = 1000;
    static constexpr size_t default_linear_treshold = 1000000;

    Math_Functions_Policy(Mode m = AUTO) : mode(m)
    {
        num_gpus = detect_num_gpus();
    }





    inline int detect_num_gpus() const
    {
        int n = omp_get_num_devices();
        return (n > 0) ? n : 0;
    }

    bool should_use_gpu(const size_t problem_size,
                        const size_t threshold,
                        const bool any_input_output_on_device)const
    {
        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            return (any_input_output_on_device) || ((num_gpus > 0) && (problem_size <= max_problem_size_for_gpu) && (problem_size >= threshold));
        }
        return false;
    }

    template <typename T>
    bool should_use_gpu(const DataBlock<T>& A,
                        const  DataBlock<T>& B,
                        const DataBlock<T>& C,
                        const size_t threshold)const
    {
        size_t problem_size = A.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(A, devicenum);
            const bool B_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(B, devicenum);
            const bool C_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(C, devicenum);
            return should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev || C_on_dev);
        }

        return false;
    }

    template <typename T>
    bool should_use_gpu( const DataBlock<T>& v1,
                         const DataBlock<T>& v2,
                         const size_t threshold)const
    {
        size_t problem_size = v1.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:

            return (num_gpus > 0);  // use cached value
        case AUTO:
            bool A_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(v1, devicenum);
            bool B_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(v2, devicenum);
            return should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev);

        }
        return false;
    }

    template <typename T>
    bool should_use_gpu( const DataBlock<T>& v1,
                         const size_t threshold)const
    {
        size_t problem_size = v1.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            bool A_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(v1, devicenum);
            return should_use_gpu(problem_size, threshold, A_on_dev);

        }
    }


};





template <typename T>
class Math_Functions
{
public:
    inline static void matrix_multiply_dot(const  DataBlock<T>& A,const  DataBlock<T>& B,  DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);
    inline static void matrix_multiply_dot_kahan( const DataBlock<T>& A, const DataBlock<T>& B,  DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);
    inline static void matrix_add(const DataBlock<T>& A,const DataBlock<T>& B, DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);
    inline static void matrix_subtract(const DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,const Math_Functions_Policy* policy=nullptr);


    inline static void matrix_multiply_vector(const  DataBlock<T>&M,const  DataBlock<T> V, DataBlock<T> C,const Math_Functions_Policy* policy=nullptr);
    inline static void matrix_multiply_vector(const  DataBlock<T>&M,const T*V,  DataBlock<T> & C, const Math_Functions_Policy* policy=nullptr);
    inline static void matrix_multiply_scalar(const   DataBlock<T>& M,const T V, DataBlock<T>& C, const Math_Functions_Policy* policy=nullptr);

    inline static void vector_multiply_scalar( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res,const Math_Functions_Policy* policy=nullptr);
    inline static void vector_add( const DataBlock<T>& vec1,  const DataBlock<T>& vec2, DataBlock<T> & res,const Math_Functions_Policy* policy=nullptr);
    inline static void vector_subtract( const DataBlock<T>& vec1, const DataBlock<T>& vec2, DataBlock<T> & res,  const Math_Functions_Policy* policy=nullptr);

    inline static T dot_product( const DataBlock<T> &vec1, const DataBlock<T> &vec2, const Math_Functions_Policy* policy=nullptr);
    inline static void cholesky_decomposition(const DataBlock<T>& A, DataBlock<T> & L, const Math_Functions_Policy* policy=nullptr);
    inline static void lu_decomposition(const DataBlock<T> &A, DataBlock<T> & L,DataBlock<T> & U, const Math_Functions_Policy* policy=nullptr);
    inline static void qr_decomposition(const DataBlock<T> &A,DataBlock<T>& Q, DataBlock<T> & R,  const Math_Functions_Policy* policy=nullptr);

    // optional default policy (initially empty = not constructed)
    inline static std::optional<Math_Functions_Policy> default_policy;

    // helper to access it with lazy init
    static const Math_Functions_Policy& get_default_policy()
    {
        if (!default_policy.has_value())
        {
            // only construct when needed
            default_policy.emplace(Math_Functions_Policy::AUTO);
        }
        return *default_policy;
    }

    static void set_default_policy(const Math_Functions_Policy& p)
    {
        default_policy = p; // assigns, overwrites if already constructed
    }

    static void reset_default_policy()
    {
        default_policy.reset(); // clear back to "uninitialized"
    }


protected:

};




template <typename T>
void Math_Functions<T>::matrix_multiply_dot( const DataBlock<T>& A,const  DataBlock<T>& B, DataBlock<T>& C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();

    if (policy.should_use_gpu(A, B, C, Math_Functions_Policy::default_cubic_treshold))
    {
        GPU_Math_Functions<T>::matrix_multiply_dot_g(A,B,C, policy.devicenum,policy.update_host);
    }
    else
        In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(A,B,C);
}


template <typename T>
void Math_Functions<T>::matrix_add( const DataBlock<T>& A,const DataBlock<T>& B,  DataBlock<T>& C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(A, B, C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions<T>::matrix_add_g(A,B,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions<T>::matrix_add_w(A,B,C);
}


template <typename T>
void Math_Functions<T>::matrix_subtract(const  DataBlock<T>& A, const DataBlock<T>& B, DataBlock<T>& C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(A, B, C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions<T>::matrix_subtract_g(A,B,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions<T>::matrix_subtract_w(A,B,C);

}


template <typename T>
void Math_Functions<T>::matrix_multiply_vector( const DataBlock<T>&M, const DataBlock<T> V, DataBlock<T> C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(M, V, C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions<T>::matrix_multiply_vector_g(M,V,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions<T>::matrix_multiply_vector_w(M,V,C);
}

template <typename T>
void Math_Functions<T>::matrix_multiply_vector(  const DataBlock<T>&M, const T*V, DataBlock<T> & C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(M,C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions<T>::matrix_multiply_vector_g(M,V,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions<T>::matrix_multiply_vector_w(M,V,C);
}




template <typename T>
void Math_Functions<T>::matrix_multiply_scalar( const  DataBlock<T>& M, const T V, DataBlock<T>& C,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(M,C, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions<T>::matrix_multiply_scalar_g(M,V,C, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions<T>::matrix_multiply_scalar_w(M,V,C);
}



template <typename T>
void Math_Functions<T>::vector_multiply_scalar( const DataBlock<T>& vec,const T scalar,DataBlock<T>& res,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(vec,res, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions<T>::matrix_multiply_scalar_g(vec,scalar,res, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions<T>::matrix_multiply_scalar_w(vec,scalar,res);
}




template <typename T>
inline void Math_Functions<T>::vector_add(  const DataBlock<T>& vec1, const DataBlock<T>& vec2, DataBlock<T> & res,const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(vec1,vec2,res, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions<T>::vector_add_g(vec1,vec2,res, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions<T>::vector_add_w(vec1,vec2,res);
}


template <typename T>
inline void Math_Functions<T>::vector_subtract(const DataBlock<T>& vec1,const DataBlock<T>& vec2, DataBlock<T> & res, const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(vec1,vec2,res, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions<T>::vector_subtract_g(vec1,vec2,res, policy.devicenum,policy.update_host);
    else
        In_Kernel_Mathfunctions<T>::vector_subtract_w(vec1,vec2,res);
}


template <typename T>
inline T Math_Functions<T>::dot_product( const DataBlock<T> &vec1, const DataBlock<T> &vec2, const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(vec1,vec2, Math_Functions_Policy::default_square_treshold))
        return GPU_Math_Functions<T>::dot_product_g(vec1,vec2, policy.devicenum);
    else
        return In_Kernel_Mathfunctions<T>::dot_product_w(vec1,vec2);
}


template <typename T>
void Math_Functions<T>::cholesky_decomposition(const DataBlock<T> & A, DataBlock<T> & L, const Math_Functions_Policy*pol)
{

    const Math_Functions_Policy &policy = (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(A,L, Math_Functions_Policy::default_square_treshold))
        GPU_Math_Functions<T>::cholesky_decomposition_g(A,L, policy.devicenum,policy.update_host,policy.initialize_output_to_zeros);
    else
    {
        In_Kernel_Mathfunctions<T>::cholesky_decomposition_w(A,L,policy.initialize_output_to_zeros);
    }

}

template <typename T>
void Math_Functions<T>::lu_decomposition(const DataBlock<T>& A, DataBlock<T> &L,DataBlock<T>& U, const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy&policy =  (pol != nullptr) ? *pol : get_default_policy();

    if (policy.should_use_gpu(A,L,U, Math_Functions_Policy::default_cubic_treshold))
        GPU_Math_Functions<T>::lu_decomposition_g(A,L,U, policy.devicenum,policy.update_host,policy.initialize_output_to_zeros);
    else
    {
        In_Kernel_Mathfunctions<T>::lu_decomposition_w(A,L,U,policy.initialize_output_to_zeros);
    }

}
// Fast QR Decomposition Algorithm for mdspan
template <typename T>
void Math_Functions<T>::qr_decomposition(const DataBlock<T>& A, DataBlock<T>& Q, DataBlock<T>& R,   const Math_Functions_Policy*pol)
{
    const Math_Functions_Policy&policy =  (pol != nullptr) ? *pol : get_default_policy();
    if (policy.should_use_gpu(A,Q,R, Math_Functions_Policy::default_cubic_treshold))
        GPU_Math_Functions<T>::qr_decomposition_g(A,Q,R, policy.devicenum,policy.update_host,policy.initialize_output_to_zeros,policy.memmapped_files);
    else
    {
        In_Kernel_Mathfunctions<T>::qr_decomposition_w(A,Q,R,policy.initialize_output_to_zeros,policy.memmapped_files);
    }
}




#endif
