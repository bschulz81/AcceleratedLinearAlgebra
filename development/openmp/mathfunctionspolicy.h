
#ifndef MATHFUNCTIONSPOLICY
#define MATHFUNCTIONSPOLICY

#include "gpu_memory_functions.h"

template<typename T>
class DataBlock;

class GPU_Memory_Functions;




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
    ptrdiff_t precision=1;
    int devicenum = omp_get_default_device();
    int num_gpus = 0;

    static constexpr ptrdiff_t max_problem_size_for_gpu = SIZE_MAX;
    static constexpr ptrdiff_t default_cubic_treshold = 256;
    static constexpr ptrdiff_t default_square_treshold = 1000;
    static constexpr ptrdiff_t default_linear_treshold = 1000000;

    Math_Functions_Policy(Mode m = AUTO) : mode(m)
    {
        num_gpus = detect_num_gpus();
    }





    inline int detect_num_gpus() const
    {
        int n = omp_get_num_devices();
        return (n > 0) ? n : 0;
    }

    bool should_use_gpu(const ptrdiff_t problem_size,
                        const ptrdiff_t threshold,
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
                        const ptrdiff_t threshold)const
    {
        ptrdiff_t problem_size = A.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = GPU_Memory_Functions::is_on_gpu(A, devicenum);
            const bool B_on_dev = GPU_Memory_Functions::is_on_gpu(B, devicenum);
            const bool C_on_dev = GPU_Memory_Functions::is_on_gpu(C, devicenum);
            return should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev || C_on_dev);
        }

        return false;
    }

    template <typename T>
    bool should_use_gpu( const DataBlock<T>& v1,
                         const DataBlock<T>& v2,
                         const ptrdiff_t threshold)const
    {
        ptrdiff_t problem_size = v1.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:

            return (num_gpus > 0);  // use cached value
        case AUTO:
            bool A_on_dev = GPU_Memory_Functions::is_on_gpu(v1, devicenum);
            bool B_on_dev = GPU_Memory_Functions::is_on_gpu(v2, devicenum);
            return should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev);

        }
        return false;
    }

    template <typename T>
    bool should_use_gpu( const DataBlock<T>& v1,
                         const ptrdiff_t threshold)const
    {
        ptrdiff_t problem_size = v1.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            bool A_on_dev = GPU_Memory_Functions::is_on_gpu(v1, devicenum);
            return should_use_gpu(problem_size, threshold, A_on_dev);

        }
    }


};






#endif
