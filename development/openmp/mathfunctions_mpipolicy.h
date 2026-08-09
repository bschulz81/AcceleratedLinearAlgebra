#ifndef  MATHFUNCTIONSMPIPOLICY
#define  MATHFUNCTIONSMPIPOLICY
#include "omp.h"
#include "datablock.h"
#include "mathfunctionspolicy.h"
#include "gpu_memory_functions.h"




class Math_MPI_Functions_Policy:public Math_Functions_Policy
{

public:
    Math_MPI_Functions_Policy(Mode m = AUTO):Math_Functions_Policy(m) {}


    using Math_Functions_Policy::should_use_gpu_elementwise;
    using Math_Functions_Policy::should_use_gpu_vector;
    using Math_Functions_Policy::should_use_gpu_matrix;
    using Math_Functions_Policy::should_use_gpu_matrix_vector;
    using Math_Functions_Policy::should_use_gpu_matrix_multiply;
    using Math_Functions_Policy::should_use_gpu_decomposition;
    using Math_Functions_Policy::should_use_gpu_sparse_matrix_vector;
    using Math_Functions_Policy::should_use_gpu_sparse_matrix_multiply;

    template<typename T>
    bool should_use_gpu_elementwise(
        const DistributedDataBlock<T>& A,
        const DistributedDataBlock<T>& B,
        const DistributedDataBlock<T>& C) const
    {
        ptrdiff_t work =
            A.pglobal_extents[0];

        size_t memory_bytes =
            sizeof(T) *
            (
                A.Dblockarray.pdatalength +
                B.Dblockarray.pdatalength +
                C.Dblockarray.pdatalength
            );

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A.Dblockarray, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(B.Dblockarray, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(C.Dblockarray, devicenum);

        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }


    template<typename T>
    bool should_use_gpu_matrix_multiply(
        const DistributedDataBlock<T>& A,
        const DistributedDataBlock<T>& B,
        const DistributedDataBlock<T>& C) const
    {
        ptrdiff_t work =
            A.pglobal_extents[0] *
            A.pglobal_extents[1] *
            B.pglobal_extents[1];

        size_t memory_bytes =
            sizeof(T) *
            (
                A.Dblockarray.pdatalength +
                B.Dblockarray.pdatalength +
                C.Dblockarray.pdatalength
            );

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A.Dblockarray, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(B.Dblockarray, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(C.Dblockarray, devicenum);

        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_matmul_threshold);
    }

    template<typename T>
    bool should_use_gpu_matrix_vector(
        const DistributedDataBlock<T>& A,
        const DistributedDataBlock<T>& x,
        const DistributedDataBlock<T>& y) const
    {
        ptrdiff_t work =
            A.pglobal_extents[0] *
            A.pglobal_extents[1];

        size_t memory_bytes =
            sizeof(T) *
            (
                A.Dblockarray.pdatalength +
                x.Dblockarray.pdatalength +
                y.Dblockarray.pdatalength
            );

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A.Dblockarray, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(x.Dblockarray, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(y.Dblockarray, devicenum);

        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }

    template<typename T>
    bool should_use_gpu_vector(
        const DistributedDataBlock<T>& x,
        const DistributedDataBlock<T>& y) const
    {
        ptrdiff_t work =
            x.pglobal_extents[0];

        size_t memory_bytes =
            sizeof(T) *
            (
                x.Dblockarray.pdatalength +
                y.Dblockarray.pdatalength
            );

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(x.Dblockarray, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(y.Dblockarray, devicenum);

        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }

    template<typename T>
    bool should_use_gpu_vector(
        const DistributedDataBlock<T>& x,
        const DistributedDataBlock<T>& y,
        const DistributedDataBlock<T>& z) const
    {
        ptrdiff_t work =
            x.pglobal_extents[0];

        size_t memory_bytes =
            sizeof(T) *
            (
                x.Dblockarray.pdatalength +
                y.Dblockarray.pdatalength+
                z.Dblockarray.pdatalength);

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(x.Dblockarray, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(y.Dblockarray, devicenum)||
             GPU_Memory_Functions::is_on_gpu(z.Dblockarray, devicenum);

        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }

    template<typename T>
    bool should_use_gpu_vector(
        const DistributedDataBlock<T>& x) const
    {
        ptrdiff_t work =
            x.pglobal_extents[0];


        size_t memory_bytes =
            sizeof(T) *
            x.Dblockarray.pdatalength;


        bool on_device =
            GPU_Memory_Functions::is_on_gpu(
                x.Dblockarray,
                devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }
  template<typename T>
    bool should_use_gpu_matrix(
        const DistributedDataBlock<T>& x) const
    {
        ptrdiff_t work =
            x.pglobal_extents[0]*x.pglobal_extents[1];


        size_t memory_bytes =
            sizeof(T) *
            x.Dblockarray.pdatalength;


        bool on_device =
            GPU_Memory_Functions::is_on_gpu(
                x.Dblockarray,
                devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }

    template<typename T>
    bool should_use_gpu_matrix(
        const DistributedDataBlock<T>& x,
        const DistributedDataBlock<T>& y
        ) const
    {
        ptrdiff_t work =
            x.pglobal_extents[0]*x.pglobal_extents[1];


        size_t memory_bytes =
            sizeof(T) *
            (x.Dblockarray.pdatalength+
            y.Dblockarray.pdatalength);


        bool on_device =
            GPU_Memory_Functions::is_on_gpu(
                x.Dblockarray,
                devicenum)||
            GPU_Memory_Functions::is_on_gpu(
                y.Dblockarray,
                devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }

    template<typename T>
    bool should_use_gpu_matrix(
        const DistributedDataBlock<T>& x,
        const DistributedDataBlock<T>& y,
        const DistributedDataBlock<T>& z)
    {
        ptrdiff_t work =
            x.pglobal_extents[0]*x.pglobal_extents[1];


        size_t memory_bytes =
            sizeof(T) *
            (x.Dblockarray.pdatalength+
            y.Dblockarray.pdatalength+
            z.Dblockarray.pdatalength);


        bool on_device =  GPU_Memory_Functions::is_on_gpu(x.Dblockarray, devicenum)||
                GPU_Memory_Functions::is_on_gpu(  y.Dblockarray, devicenum)||
                GPU_Memory_Functions::is_on_gpu(  z.Dblockarray,   devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }

    template<typename T>
    bool should_use_gpu_decomposition(
        const DistributedDataBlock<T>& A) const
    {
        ptrdiff_t n =
            A.pglobal_extents[0];


        ptrdiff_t work =
            n*n*n;


        size_t memory_bytes =
            sizeof(T) *
            A.Dblockarray.pdatalength;


        bool on_device =
            GPU_Memory_Functions::is_on_gpu(
                A.Dblockarray,
                devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_decomposition_threshold);
    }


protected:
    inline bool should_use_gpu_work(
        ptrdiff_t work,
        size_t memory_bytes,
        bool data_on_device,
        ptrdiff_t threshold) const
    {
        if(!rank_can_use_gpu())
            return false;


        return Math_Functions_Policy::should_use_gpu_work(
                   work,
                   memory_bytes,
                   data_on_device,
                   threshold);
    }


    bool rank_can_use_gpu() const
    {
        return devicenum >= 0;
    }

private:


};


class Math_MPI_RecursiveMultiplication_Policy
    : public Math_MPI_Functions_Policy
{

public:

using Math_MPI_Functions_Policy::should_use_gpu_matrix_multiply;
    using Math_MPI_Functions_Policy::should_use_gpu_matrix_vector;
    using Math_MPI_Functions_Policy::should_use_gpu_elementwise;
    using Math_MPI_Functions_Policy::should_use_gpu_vector;
    using Math_MPI_Functions_Policy::should_use_gpu_matrix;
    using Math_MPI_Functions_Policy::should_use_gpu_decomposition;
    using Math_MPI_Functions_Policy::should_use_gpu_sparse_matrix_vector;
    using Math_MPI_Functions_Policy::should_use_gpu_sparse_matrix_multiply;

    bool usempi=true;

    ptrdiff_t size_to_stop_recursion = 64;

    ptrdiff_t gpu_activation_depth = 1;
    enum Matrix_Multiplication_Algorithm
    {
        Naive=0,
        Strassen=1,
        WinogradVariant=2
    } algorithm_version=Naive;

    enum Listener_Commands
    {
        StartStrassen=1,
        StartWinogradVariant=2,
        End_Listener=3
    };

Math_MPI_RecursiveMultiplication_Policy(bool busempi=true,Math_MPI_Functions_Policy::Mode m=AUTO,Matrix_Multiplication_Algorithm algorithm=Matrix_Multiplication_Algorithm::Naive):
                                       Math_MPI_Functions_Policy(m),usempi(busempi),algorithm_version(algorithm)
                                       {}




template<typename T>
bool should_use_gpu_matrix_multiply(
    ptrdiff_t rowsA,
    ptrdiff_t colsA,
    ptrdiff_t colsB,
    bool already_on_gpu = false,
    size_t extra_memory_bytes = 0) const
{
    ptrdiff_t work =
        rowsA * colsA * colsB;

    size_t memory_bytes =
        sizeof(T) *
        (
            rowsA * colsA +
            colsA * colsB +
            rowsA * colsB
        )
        + extra_memory_bytes;

    return should_use_gpu_work(
        work,
        memory_bytes,
        already_on_gpu,
        gpu_matmul_threshold);
}

  bool should_use_mpi_for_recursion(
        int mpi_rank,
        int mpi_size,
        int number_of_children = 7) const
    {
        if(!usempi) return false;

        int first_child =
            mpi_rank * number_of_children + 1;

        return first_child < mpi_size;
    }

    bool should_use_recursive_multiplication(
        ptrdiff_t problem_size) const
    {
        return problem_size > size_to_stop_recursion;
    }



    /*
       Estimate temporary memory needed by one Winograd level.
       This corresponds to the allocations:

       S1-S4 : 4 * (n/2*m/2)
       S5-S8 : 4 * (m/2*p/2)
       M1-M7 : 7 * (n/2*p/2)
    */
    template<typename T>
    size_t winograd_workspace_bytes(
        ptrdiff_t n,
        ptrdiff_t m,
        ptrdiff_t p) const
    {
        size_t elements =
            4 * (n/2) * (m/2) +
            4 * (m/2) * (p/2) +
            7 * (n/2) * (p/2);

        return sizeof(T) * elements;
    }




    template<typename T>
    bool should_use_gpu_winograd_start(
        const DataBlock<T>& A,
        const DataBlock<T>& B,
        const DataBlock<T>& C) const
    {
        ptrdiff_t n = A.extent(0);
        ptrdiff_t m = A.extent(1);
        ptrdiff_t p = B.extent(1);


        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(B,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(C,devicenum);



#if defined(Unified_Shared_Memory)
        return should_use_gpu_matrix_multiply(A,B,C);
#else

        size_t input_output_memory =
            sizeof(T) *
            (
                A.datalength() +
                B.datalength() +
                C.datalength()
            );


        size_t workspace_memory =
            winograd_workspace_bytes<T>(n,m,p);


        size_t total_memory =
            input_output_memory +
            workspace_memory;


        ptrdiff_t work =
            n*m*p;


        if(mode == GPU_ONLY)
            return rank_can_use_gpu();


        if(mode == CPU_ONLY)
            return false;


        if(on_device)
            return true;


        return rank_can_use_gpu() &&
               total_memory <= max_gpu_memory_bytes &&
               work >= gpu_matmul_threshold;

#endif
    }

    template<typename T>
    bool should_use_naive_algorithm(
        const DataBlock<T>& A,
        const DataBlock<T>& B,
        const DataBlock<T>& C,
        bool already_on_gpu) const
    {
        ptrdiff_t n = A.extent(0);
        ptrdiff_t m = A.extent(1);
        ptrdiff_t p = B.extent(1);


        // Always stop recursion at small  matrices whose extents cant be divided by 2
        if((n <= size_to_stop_recursion ||
                m <= size_to_stop_recursion ||
                p <= size_to_stop_recursion)||((n%2!=0) || (m%2!=0) || (p%2!=0) ||m<=2 || n<=2 || p<=2))
        {
            return true;
        }



        return false;
    }
};

struct Math_MPI_Decomposition_Policy : public Math_MPI_RecursiveMultiplication_Policy
{
public:
    Math_MPI_Decomposition_Policy(bool pusempi, Math_Functions_Policy::Mode m):
        Math_MPI_RecursiveMultiplication_Policy(pusempi,m){}

    ptrdiff_t step_size=0;

    using Math_MPI_RecursiveMultiplication_Policy::Math_MPI_RecursiveMultiplication_Policy;



};



enum class SummaGridPolicy
{
    DenseOnly,
    Compatible,
    LoadBalanced
};
#endif
