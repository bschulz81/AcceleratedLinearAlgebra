#ifndef MATHFUNCTIONSPOLICY
#define MATHFUNCTIONSPOLICY

#include "gpu_memory_functions.h"

template<typename T>
class DataBlock;

class GPU_Memory_Functions;




struct DeviceInfo
{
    int dev_id;
    int num_teams;
    int threads_per_team;
};

// Query function
inline DeviceInfo query_device_team_thread_counts(int dev)
{
    DeviceInfo info;
    info.dev_id = dev;
    info.num_teams = 0;
    info.threads_per_team = 0;

    #pragma omp target map(from: info) device(dev)
    {
        #pragma omp teams
        {
            if (omp_get_team_num() == 0)
                info.num_teams = omp_get_num_teams();

            #pragma omp parallel
            {
                if (omp_get_thread_num() == 0)
                    info.threads_per_team = omp_get_num_threads();
            }
        }
    }

    return info;
}





class Math_Functions_Policy
{
public:

    enum Mode
    {
        CPU_ONLY,
        GPU_ONLY,
        AUTO
    } mode = AUTO;


    bool update_host = true;
    bool memmapped_files = false;
    bool initialize_output_to_zeros = true;

    int accumulation_precision = 1;

    int devicenum = omp_get_default_device();
    int num_gpus = omp_get_num_devices();


    ptrdiff_t  max_gpu_memory_bytes = SIZE_MAX;

    ptrdiff_t  gpu_linear_threshold = 1000000;
    ptrdiff_t  gpu_matmul_threshold = 256*256*256;
    ptrdiff_t  gpu_decomposition_threshold = 256*256*256;


    Math_Functions_Policy(Mode m = AUTO)
        : mode(m)
    {
        num_gpus = detect_num_gpus();
    }


    bool should_use_gpu_work(
        ptrdiff_t work,
        size_t memory_bytes,
        bool data_on_device,
        ptrdiff_t threshold) const
    {
        switch(mode)
        {
        case CPU_ONLY:
            return false;

        case GPU_ONLY:
            return true;

        case AUTO:
            if(data_on_device)
                return true;

            return (memory_bytes <= max_gpu_memory_bytes) &&
                   (work >= threshold);
        }

        return false;
    }

    template<typename T>
    bool should_use_gpu_elementwise(
        const DataBlock<T>& A,
        const DataBlock<T>& B,
        const DataBlock<T>& C) const
    {
        ptrdiff_t work = A.datalength();

        size_t memory_bytes =
            sizeof(T) *
            (A.datalength() +
             B.datalength() +
             C.datalength());

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(B, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(C, devicenum);

        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }
 template<typename T>
    bool should_use_gpu_elementwise(
        const DataBlock<T>& A,
        const DataBlock<T>& C) const
    {
        ptrdiff_t work = A.datalength();

        size_t memory_bytes =
            sizeof(T) *
            (A.datalength() +
             C.datalength());

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A, devicenum) ||
            GPU_Memory_Functions::is_on_gpu(C, devicenum);

        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }

    template<typename T>
    bool should_use_gpu_elementwise(
        const DataBlock<T>& C) const
    {
        ptrdiff_t work = C.datalength();

        size_t memory_bytes =sizeof(T) *C.datalength();

        bool on_device =GPU_Memory_Functions::is_on_gpu(C, devicenum);

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
            GPU_Memory_Functions::is_on_gpu(x.Dblockarray,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(y.Dblockarray,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }


    template<typename T>
    bool should_use_gpu_matrix_vector(
        const DataBlock<T>& A,
        const DataBlock<T>& x,
        const DataBlock<T>& y) const
    {
        ptrdiff_t work =
            A.extent(0) *
            A.extent(1);

        size_t memory_bytes =
            sizeof(T) *
            (A.datalength() +
             x.datalength() +
             y.datalength());


        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(x,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(y,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_linear_threshold);
    }

    template<typename T>
    bool should_use_gpu_matrix_multiply(
        const DataBlock<T>& A,
        const DataBlock<T>& B,
        const DataBlock<T>& C) const
    {
        ptrdiff_t work =
            A.extent(0) *
            A.extent(1) *
            B.extent(1);

        size_t memory_bytes =
            sizeof(T) *
            (A.datalength() +
             B.datalength() +
             C.datalength());

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(B,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(C,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_matmul_threshold);
    }


     template<typename T>
    bool should_use_gpu_matrix(
        const DataBlock<T>& A,
        const DataBlock<T>& B,
        const DataBlock<T>& C) const
    {
        ptrdiff_t work =
            A.extent(0) *
            A.extent(1) ;

        size_t memory_bytes =
            sizeof(T) *
            (A.datalength() +
             B.datalength() +
             C.datalength());

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(B,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(C,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_matmul_threshold);
    }

       template<typename T>
    bool should_use_gpu_vector(
        const DataBlock<T>& A,
        const DataBlock<T>& B,
        const DataBlock<T>& C) const
    {
        ptrdiff_t work =
            A.extent(0);

        size_t memory_bytes =
            sizeof(T) *
            (A.datalength() +
             B.datalength() +
             C.datalength());

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(B,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(C,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_matmul_threshold);
    }
        template<typename T>

    bool should_use_gpu_vector(
        const DataBlock<T>& A,
        const DataBlock<T>& C) const
    {
        ptrdiff_t work =
            A.extent(0);

        size_t memory_bytes =
            sizeof(T) *
            (A.datalength() +
             C.datalength());

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(C,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_matmul_threshold);
    }
       template<typename T>
        bool should_use_gpu_vector(
        const DataBlock<T>& A) const
    {
        ptrdiff_t work =
            A.extent(0);

        size_t memory_bytes =
            sizeof(T) *
            (A.datalength());

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_matmul_threshold);
    }

     template<typename T>
    bool should_use_gpu_matrix(
        const DataBlock<T>& A,
        const DataBlock<T>& C) const
    {
        ptrdiff_t work =
            A.extent(0) *
            A.extent(1) ;

        size_t memory_bytes =
            sizeof(T) *
            (A.datalength() +
             C.datalength());

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A,devicenum) ||
            GPU_Memory_Functions::is_on_gpu(C,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_matmul_threshold);
    }

     template<typename T>
    bool should_use_gpu_matrix(
        const DataBlock<T>& A) const
    {
        ptrdiff_t work =
            A.extent(0) *
            A.extent(1) ;

        size_t memory_bytes =
            sizeof(T) *
            (A.datalength() );

        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_matmul_threshold);
    }


    template<typename T>
    bool should_use_gpu_decomposition(
        const DataBlock<T>& A) const
    {
        ptrdiff_t work =
            A.extent(0) *
            A.extent(1) *
            A.extent(0);


        size_t memory_bytes =
            sizeof(T) *
            A.datalength();


        bool on_device =
            GPU_Memory_Functions::is_on_gpu(A,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_decomposition_threshold);
    }
template<typename T>
bool should_use_gpu_sparse_matrix_vector(
    const BlockedDataView<T>& A,
    const DataBlock<T>& x,
    const DataBlock<T>& y) const
{
    ptrdiff_t work =
        A.number_of_blocks() *
        A.block_volume();

    size_t memory_bytes =
        sizeof(T) *
        (
            A.get_datablock().datalength() +
            x.datalength() +
            y.datalength()
        );

    bool on_device =
        GPU_Memory_Functions::is_on_gpu(
            A.get_datablock(),
            devicenum)
        ||
        GPU_Memory_Functions::is_on_gpu(
            x,
            devicenum)
        ||
        GPU_Memory_Functions::is_on_gpu(
            y,
            devicenum);

    return should_use_gpu_work(
        work,
        memory_bytes,
        on_device,
        gpu_linear_threshold);
}
    template<typename T>
    bool should_use_gpu_sparse_matrix_multiply(
        const BlockedDataView<T>& A,
        const BlockedDataView<T>& B,
        const DataBlock<T>& C) const
    {
        ptrdiff_t work =
            A.number_of_blocks() *
            B.number_of_blocks() *
            std::max(A.block_volume(),
                     B.block_volume());


        size_t memory_bytes =
            sizeof(T) *
            (
                A.get_datablock().datalength() +
                B.get_datablock().datalength() +
                C.datalength()
            );


        bool on_device =
            GPU_Memory_Functions::is_on_gpu(
                A.get_datablock(),devicenum)
            ||
            GPU_Memory_Functions::is_on_gpu(
                B.get_datablock(),devicenum)
            ||
            GPU_Memory_Functions::is_on_gpu(
                C,devicenum);


        return should_use_gpu_work(
                   work,
                   memory_bytes,
                   on_device,
                   gpu_matmul_threshold);
    }



private:

    inline int detect_num_gpus() const
    {
        int n = omp_get_num_devices();
        return (n > 0) ? n : 0;
    }


    inline bool gpu_available() const
    {
        return num_gpus > 0;
    }



};





#endif
