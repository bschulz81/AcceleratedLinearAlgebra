
#ifndef  MATHFUNCTIONSMPIPOLICY
#define  MATHFUNCTIONSMPIPOLICY
#include "omp.h"
#include "mathfunctionspolicy.h"
#include "gpu_memory_functions.h"
class Math_MPI_Functions_Policy : public Math_Functions_Policy, public MPI_Policy
{
public:
    bool allow_gpu_sharing = true;

    Math_MPI_Functions_Policy(Mode m = AUTO,bool gpu_sharing=true,bool mpi=true,MPI_Comm comm=MPI_COMM_WORLD)
        : Math_Functions_Policy(m), MPI_Policy (mpi, comm),    allow_gpu_sharing(gpu_sharing)


    {
        // Use cached num_gpus from base class
        if(mpi_enabled)
        {
            if (num_gpus > 0)
            {
                allow_gpu_sharing=gpu_sharing;
                if (allow_gpu_sharing)
                    devicenum = mpi_rank % num_gpus;  // shared mode
                else if (mpi_rank < num_gpus)
                    devicenum = mpi_rank;             // exclusive mode
                else
                    devicenum = -INT_MAX;                   // CPU fallback
            }
            else
            {
                devicenum = -INT_MAX; // no GPU available
            }
        }
    }

    bool rank_can_use_gpu() const
    {
        return ((mpi_enabled) && (devicenum >= 0));
    }

    bool should_use_gpu(const ptrdiff_t problem_size,
                        const ptrdiff_t threshold,
                        const bool any_input_output_on_device)const
    {
        if (!Math_Functions_Policy::should_use_gpu(problem_size, threshold,any_input_output_on_device))
            return false;
        else
        {
            if(mpi_enabled)
                return rank_can_use_gpu();
            else
            {
                return true;
            }

        }
    }

    template <typename T>
    bool should_use_gpu(const DistributedDataBlock<T>& A,
                        const DistributedDataBlock<T>& B,
                        const  DistributedDataBlock<T>& C,
                        const ptrdiff_t threshold)const
    {
        const ptrdiff_t problem_size = A.Dblockarray.pdatalength;

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = A.Dblockarray.pdata_is_devptr ;
            const bool B_on_dev = B.Dblockarray.pdata_is_devptr ;
            const bool C_on_dev = C.Dblockarray.pdata_is_devptr;
            if(A_on_dev  || B_on_dev ||C_on_dev)
                return true;


            return this->should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev || C_on_dev);
        }
        return false;
    }


    template <typename T>
    bool should_use_gpu(const DataBlock<T>& A,
                        const DataBlock<T>& B,
                        const  DataBlock<T>& C,
                        const ptrdiff_t threshold)const
    {
        const ptrdiff_t problem_size = A.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = A.dpconfig.data_is_devptr;
            const bool B_on_dev = B.dpconfig.data_is_devptr;
            const bool C_on_dev = C.dpconfig.data_is_devptr;
            if(A_on_dev|| C_on_dev|| B_on_dev) return true;


            return this->should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev || C_on_dev);
        }
        return false;
    }

    template <typename T>
    bool should_use_gpu(const DataBlock<T>& v1,
                        const DataBlock<T>& v2,
                        const ptrdiff_t threshold)const
    {
        const ptrdiff_t problem_size = v1.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = v1.dpconfig.data_is_devptr;
            const bool C_on_dev = v2.dpconfig.data_is_devptr;
            if(A_on_dev||C_on_dev) return true;

            return this->should_use_gpu(problem_size, threshold, A_on_dev  || C_on_dev);

        }
    }

    template <typename T>
    bool should_use_gpu(const DistributedDataBlock<T>& v1,
                        const DistributedDataBlock<T>& v2,
                        const ptrdiff_t threshold)const
    {
        const ptrdiff_t problem_size = v1.Dblockarray.pdatalength;

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = v1.Dblockarray.pdata_is_devptr;
            const bool B_on_dev = v2.Dblockarray.pdata_is_devptr;

            if(A_on_dev||B_on_dev) return true;

            return this->should_use_gpu(problem_size, threshold, A_on_dev  || B_on_dev);

        }
        return false;
    }


    template <typename T>
    bool should_use_gpu(const DataBlock<T>& v1,
                        const ptrdiff_t threshold)const
    {
        const ptrdiff_t problem_size = v1.datalength();
        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = v1.dpconfig.data_is_devptr;
            if(A_on_dev) return true;
            return this->should_use_gpu(problem_size, threshold, A_on_dev );

        }
        return false;
    }

    template <typename T>
    bool should_use_gpu(const DistributedDataBlock<T>& v1,
                        const ptrdiff_t threshold)const
    {
        const ptrdiff_t problem_size = v1.Dblockarray.pdatalength;
        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = v1.Dblockarray.pdata_is_devptr ;
            if(A_on_dev) return true;
            return this->should_use_gpu(problem_size, threshold, A_on_dev );

        }
        return false;
    }




};


struct Math_MPI_RecursiveMultiplication_Policy : public Math_MPI_Functions_Policy
{
public:

    enum Listener_Commands
    {
        Strassen=1,
        WinogradVariant=2,
        End_Listener=3
    };

    ptrdiff_t size_to_stop_recursion = 64;  // below this size: stop recursion

    using Math_MPI_Functions_Policy::Math_MPI_Functions_Policy;

    bool should_use_mpi_for_recursion(ptrdiff_t num_subcalls) const
    {
        if (!this->mpi_enabled)
            return false;
        int myrank=0;
        MPI_Comm_rank(this->comm, &myrank);
        return std::abs(this->mpi_size) >= pow(num_subcalls,myrank+1);
    }


    bool should_use_recursion(ptrdiff_t problem_size) const
    {
        if (problem_size <= size_to_stop_recursion)
            return false; // base case → naive CPU multiply
        else
            return true;
    }


    bool should_use_gpu(const ptrdiff_t problem_size,
                        const ptrdiff_t threshold,
                        const bool any_input_output_on_device,
                        const ptrdiff_t num_subcalls)const
    {
        if (!should_use_mpi_for_recursion(num_subcalls))
        {
            // Not enough ranks to distribute → maybe still use GPU locally
            return Math_Functions_Policy::should_use_gpu(problem_size, threshold, any_input_output_on_device);
        }

        // Enough ranks → allow GPU if mapping allows it
        return Math_MPI_Functions_Policy::should_use_gpu(problem_size, threshold, any_input_output_on_device);
    }


    template <typename T>
    bool should_use_gpu(const DataBlock<T>& A,
                        const DataBlock<T>& B,
                        const DataBlock<T>& C,
                        const ptrdiff_t threshold,
                        const ptrdiff_t num_subcalls)const
    {
        ptrdiff_t problem_size = A.datalength();

        bool A_on_dev = GPU_Memory_Functions::is_on_gpu(A, devicenum);
        bool B_on_dev = GPU_Memory_Functions::is_on_gpu(B, devicenum);
        bool C_on_dev = GPU_Memory_Functions::is_on_gpu(C, devicenum);

        if(A_on_dev||B_on_dev||C_on_dev) return true;

        return should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev || C_on_dev, num_subcalls);
    }

    template <typename T>
    bool should_use_gpu(const DataBlock<T>& v1,
                        const DataBlock<T>& v2,
                        const ptrdiff_t threshold,
                        const ptrdiff_t num_subcalls)const
    {
        const ptrdiff_t problem_size = v1.datalength();

        bool v1_on_dev = GPU_Memory_Functions::is_on_gpu(v1, devicenum);
        bool v2_on_dev = GPU_Memory_Functions::is_on_gpu(v2, devicenum);
        if(v1_on_dev||v1_on_dev) return true;

        return should_use_gpu(problem_size, threshold, v1_on_dev || v2_on_dev,num_subcalls);

    }

    template <typename T>
    bool should_use_gpu(const DataBlock<T>& v1,
                        ptrdiff_t threshold,ptrdiff_t num_subcalls)
    {
        const ptrdiff_t problem_size = v1.datalength();

        const bool v1_on_dev = GPU_Memory_Functions::is_on_gpu(v1, devicenum);
        if(v1_on_dev) return true;
        return should_use_gpu(problem_size, threshold, v1_on_dev,num_subcalls);

    }
};


struct Math_MPI_Decomposition_Policy : public Math_MPI_RecursiveMultiplication_Policy
{
public:
    enum Matrix_Multiplication_Algorithm
    {
        Naive=0,
        Strassen=1,
        WinogradVariant=2
    } algorithm_version=Naive;


    ptrdiff_t step_size=0;

    using Math_MPI_RecursiveMultiplication_Policy::Math_MPI_RecursiveMultiplication_Policy;

    Math_MPI_Decomposition_Policy(
        Mode m,
        bool mpi,
        bool sharing,
        Matrix_Multiplication_Algorithm algo,
        ptrdiff_t step = 0)
        : Math_MPI_RecursiveMultiplication_Policy(m, mpi, sharing),
          algorithm_version(algo),
          step_size(step)
    {}

};




class DataBlock_MPI_Functions;

enum class SummaGridPolicy
{
    DenseOnly,        // every node gets at least one block
    Compatible,       // every node gets either at least one or several blocks, or at maximum one or zero blocks
    LoadBalanced      // among compatible grids choose best balance
};
#endif
