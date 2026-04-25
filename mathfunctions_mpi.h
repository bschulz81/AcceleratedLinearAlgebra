#ifndef MATHFUNCTIONS_MPI
#define MATHFUNCTIONS_MPI

#include "mpi.h"
#include "datablock.h"
#include "mdspan_omp.h"
#include "mdspan_data.h"
#include <math.h>

#include "datablock_host_memory_functions.h"
#include "datablock_gpu_memory_functions.h"
#include "datablock_mpifunctions.h"
#include "inkernel_mathfunctions.h"



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
                    devicenum = -1;                   // CPU fallback
            }
            else
            {
                devicenum = -1; // no GPU available
            }
        }
    }

    bool rank_can_use_gpu() const
    {
        return ((mpi_enabled) && (devicenum >= 0));
    }

    bool should_use_gpu(const size_t problem_size,
                        const size_t threshold,
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
                        const size_t threshold)const
    {
        const size_t problem_size = A.pdatalength;

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = A.pdpdata_is_devptr ;
            const bool B_on_dev = B.pdpdata_is_devptr ;
            const bool C_on_dev = C.pdpdata_is_devptr;
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
                        const size_t threshold)const
    {
        const size_t problem_size = A.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = A.dpdata_is_devptr;
            const bool B_on_dev = B.dpdata_is_devptr;
            const bool C_on_dev = C.dpdata_is_devptr;
            if(A_on_dev|| C_on_dev|| B_on_dev) return true;


            return this->should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev || C_on_dev);
        }
        return false;
    }

    template <typename T>
    bool should_use_gpu(const DataBlock<T>& v1,
                        const DataBlock<T>& v2,
                        const size_t threshold)const
    {
        const size_t problem_size = v1.datalength();

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = v1.dpdata_is_devptr;
            const bool C_on_dev = v2.dpdata_is_devptr;
            if(A_on_dev||C_on_dev) return true;

            return this->should_use_gpu(problem_size, threshold, A_on_dev  || C_on_dev);

        }
    }

    template <typename T>
    bool should_use_gpu(const DistributedDataBlock<T>& v1,
                        const DistributedDataBlock<T>& v2,
                        const size_t threshold)const
    {
        const size_t problem_size = v1.pdatalength;

        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = v1.dpdata_is_devptr;
            const bool B_on_dev = v2.dpdata_is_devptr;

            if(A_on_dev||B_on_dev) return true;

            return this->should_use_gpu(problem_size, threshold, A_on_dev  || B_on_dev);

        }
    }


    template <typename T>
    bool should_use_gpu(const DataBlock<T>& v1,
                        const size_t threshold)const
    {
        const size_t problem_size = v1.datalength();
        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = v1.dpdata_is_devptr;
            if(A_on_dev) return true;
            return this->should_use_gpu(problem_size, threshold, A_on_dev );

        }
    }

    template <typename T>
    bool should_use_gpu(const DistributedDataBlock<T>& v1,
                        const size_t threshold)const
    {
        const size_t problem_size = v1.pdatalength;
        switch (mode)
        {
        case CPU_ONLY:
            return false;
        case GPU_ONLY:
            return (num_gpus > 0);  // use cached value
        case AUTO:
            const bool A_on_dev = v1.dpdata_is_devptr ;
            if(A_on_dev) return true;
            return this->should_use_gpu(problem_size, threshold, A_on_dev );

        }
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

    size_t size_to_stop_recursion = 64;  // below this size: stop recursion

    using Math_MPI_Functions_Policy::Math_MPI_Functions_Policy;

    bool should_use_mpi_for_recursion(size_t num_subcalls) const
    {
        if (!this->mpi_enabled)
            return false;
        int myrank=0;
        MPI_Comm_rank(this->comm, &myrank);
        return std::abs(this->mpi_size) >= pow(num_subcalls,myrank+1);
    }


    bool should_use_recursion(size_t problem_size) const
    {
        if (problem_size <= size_to_stop_recursion)
            return false; // base case → naive CPU multiply
        else
            return true;
    }


    bool should_use_gpu(const size_t problem_size,
                        const size_t threshold,
                        const bool any_input_output_on_device,
                        const size_t num_subcalls)const
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
                        const size_t threshold,
                        const size_t num_subcalls)const
    {
        size_t problem_size = A.datalength();

        bool A_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(A, devicenum);
        bool B_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(B, devicenum);
        bool C_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(C, devicenum);

        if(A_on_dev||B_on_dev||C_on_dev) return true;

        return should_use_gpu(problem_size, threshold, A_on_dev || B_on_dev || C_on_dev, num_subcalls);
    }

    template <typename T>
    bool should_use_gpu(const DataBlock<T>& v1,
                        const DataBlock<T>& v2,
                        const size_t threshold,
                        const size_t num_subcalls)const
    {
        const size_t problem_size = v1.datalength();

        bool v1_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(v1, devicenum);
        bool v2_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(v2, devicenum);
        if(v1_on_dev||v1_on_dev) return true;

        return should_use_gpu(problem_size, threshold, v1_on_dev || v2_on_dev,num_subcalls);

    }

    template <typename T>
    bool should_use_gpu(const DataBlock<T>& v1,
                        size_t threshold,size_t num_subcalls)
    {
        const size_t problem_size = v1.datalength();

        const bool v1_on_dev = DataBlock_GPU_Memory_Functions<T>::is_on_gpu(v1, devicenum);
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


    size_t step_size=0;

    using Math_MPI_RecursiveMultiplication_Policy::Math_MPI_RecursiveMultiplication_Policy;

    // New constructor that also sets the algorithm
    Math_MPI_Decomposition_Policy(
        Mode m,
        bool mpi,
        bool sharing,
        Matrix_Multiplication_Algorithm algo,
        size_t step = 0)
        : Math_MPI_RecursiveMultiplication_Policy(m, mpi, sharing),
          algorithm_version(algo),
          step_size(step)
    {}

};



template<typename T>
class DataBlock_MPI_Functions;




using namespace std;


template <typename T>
class Math_Functions_MPI: public Math_Functions<T>
{
public:


    inline static void strassen_multiply(const DataBlock<T> &aA, const DataBlock<T> &aB,DataBlock<T>& aC, const Math_MPI_RecursiveMultiplication_Policy *par=nullptr);

    inline static void winograd_multiply(const DataBlock<T> &aA,const DataBlock<T> &aB,DataBlock<T>& aC, const Math_MPI_RecursiveMultiplication_Policy *par=nullptr);

    inline static void cholesky_decomposition(const DataBlock<T>& aA, DataBlock<T> & aL,  Math_MPI_Decomposition_Policy *par=nullptr);

    inline static void lu_decomposition(const DataBlock<T> &aA, DataBlock<T> & aL,DataBlock<T> & aU, Math_MPI_Decomposition_Policy *par=nullptr);

    inline static void qr_decomposition(const DataBlock<T> &aA,DataBlock<T>& aQ, DataBlock<T> & aR,    Math_MPI_Decomposition_Policy *par=nullptr);

    inline static void MPI_recursive_multiplication_helper( const Math_MPI_RecursiveMultiplication_Policy*par=nullptr);
    inline static void MPI_recursion_helper_end(MPI_Comm pcomm);
    inline static bool SUMMA_Distributed(const DistributedDataBlock<T>& A,  const DistributedDataBlock<T>& B,  DistributedDataBlock<T>& C,   const Math_MPI_Functions_Policy* pol = nullptr);
    inline static bool Matrix_Vector_multiply_Distributed( const DistributedDataBlock<T>& A,  const DistributedDataBlock<T>& x,  DistributedDataBlock<T>& y,   const Math_MPI_Functions_Policy* pol=nullptr);

protected:
    inline static void strassen_multiply_h(const DataBlock<T> &aA,const DataBlock<T> &aB,DataBlock<T>& aC,bool ongpu, bool separate_device_memory, const Math_MPI_RecursiveMultiplication_Policy &par);

    inline static void winograd_multiply_h(const DataBlock<T> &aA,const DataBlock<T> &aB,DataBlock<T>& aC,bool ongpu, bool separate_device_memory, const Math_MPI_RecursiveMultiplication_Policy &par);

    inline static void cholesky_decomposition_h(const DataBlock<T>& aA, DataBlock<T> & aL,  Math_MPI_Decomposition_Policy &par);

    inline static void lu_decomposition_h(const DataBlock<T> &aA, DataBlock<T> & aL,DataBlock<T> & aU, Math_MPI_Decomposition_Policy &par);

    inline static void qr_decomposition_h(const DataBlock<T> &aA,DataBlock<T>& aQ, DataBlock<T> & aR,    Math_MPI_Decomposition_Policy &par);



    inline static std::optional<Math_MPI_Decomposition_Policy> default_policy;



    static const Math_MPI_Decomposition_Policy& get_default_policy()
    {
        if (!default_policy.has_value())
        {

            default_policy.emplace(Math_Functions_Policy::AUTO);
        }
        return *default_policy;
    }

    static void set_default_policy(const Math_MPI_Decomposition_Policy& p)
    {
        default_policy = p;
    }

    static void reset_default_policy()
    {
        default_policy.reset();
    }
};



template <typename T>
bool Math_Functions_MPI<T>::SUMMA_Distributed(
    const DistributedDataBlock<T>& A,
    const DistributedDataBlock<T>& B,
    DistributedDataBlock<T>& C,
    const Math_MPI_Functions_Policy* pol)
{
    const Math_MPI_Functions_Policy policy =
        (pol != nullptr) ? *pol : get_default_policy();
    if(!policy.mpi_enabled)
        return false;
    if (A.pctx->comm == MPI_COMM_NULL) return false;
    if (B.pctx->comm == MPI_COMM_NULL) return false;
    if (C.pctx->comm == MPI_COMM_NULL) return false;
    if(A.pglobal_extents[1] != B.pglobal_extents[0])
        return false;
    if(A.pglobal_extents[0] != C.pglobal_extents[0])
        return false;
    if(B.pglobal_extents[1] != C.pglobal_extents[1])
        return false;
    if(A.pblock_extents[1] != B.pblock_extents[0])
        return false;
    if(A.pblock_extents[0] != C.pblock_extents[0])
        return false;
    if(B.pblock_extents[1] != C.pblock_extents[1])
        return false;



    MPI_Comm comma = C.pctx->comm;
    int rank;
    MPI_Comm_rank(comma, &rank);


    size_t blocknumber=C.plocal_blocknumber;
    size_t maxnumber,minnumber;

    MPI_Allreduce(&blocknumber, &maxnumber, 1,
                  mpi_get_type<size_t>(), MPI_MAX, comma);
    MPI_Allreduce(&blocknumber, &minnumber, 1, mpi_get_type<size_t>(), MPI_MIN, comma);

    if(maxnumber<=1)
    {
        int coords[2];
        MPI_Cart_coords(comma, rank, 2, coords);
        int my_row = coords[0];
        int my_col = coords[1];
        const size_t Pr = A.pctx->dims[0];
        const size_t Pc = A.pctx->dims[1];
        const size_t br = A.pblock_extents[0];
        const size_t bk = A.pblock_extents[1];
        const size_t bc = B.pblock_extents[1];
        const size_t M = A.pglobal_extents[0];
        const size_t N = B.pglobal_extents[1];
        const size_t Ktot = A.pglobal_extents[1];
        const size_t grid_r = (M + br - 1) / br;
        const size_t grid_c = (N + bc - 1) / bc;
        const size_t grid_k = (Ktot + bk - 1) / bk;

        MPI_Comm row_comm, col_comm;
        MPI_Comm_split(comma, my_row, my_col, &row_comm);
        MPI_Comm_split(comma, my_col, my_row, &col_comm);
        size_t max_A = br * bk;
        size_t max_B = bk * bc;
        bool ongpu=policy.should_use_gpu(A,B,C,Math_Functions_Policy::default_cubic_treshold);
        bool memmap=policy.memmapped_files;
        int devnum=policy.devicenum;
        if(A.pdpdata_is_devptr&& A.pdevptr_devicenum!=devnum) return false;
        if(B.pdpdata_is_devptr&& B.pdevptr_devicenum!=devnum) return false;
        if(C.pdpdata_is_devptr&& C.pdevptr_devicenum!=devnum) return false;
        if(A.pdevptr_devicenum!=B.pdevptr_devicenum ||A.pdevptr_devicenum!=C.pdevptr_devicenum) return false;
        T* A_buf;
        T* B_buf;
        if(max_A>0)
            DataBlock_MPI_Functions<T>::alloc_helper2(memmap,ongpu,devnum,max_A,A_buf);
        if(max_B>0)
            DataBlock_MPI_Functions<T>::alloc_helper2(memmap,ongpu,devnum,max_B,B_buf);

        T* adata=A.pdata;
        if(ongpu)
        {
            if(A.plocal_blocknumber > 0)
            {
                if(!A.pdpdata_is_devptr)
                {
                    adata=(T*) omp_target_alloc(sizeof(T)*A.pdatalength,devnum);
                    omp_target_memcpy(adata,A.pdata,sizeof(T)*A.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }
        T* bdata=B.pdata;
        if(ongpu)
        {
            if (B.plocal_blocknumber > 0)
            {
                if(!B.pdpdata_is_devptr)
                {
                    bdata=(T*) omp_target_alloc(sizeof(T)*B.pdatalength,devnum);
                    omp_target_memcpy(bdata,B.pdata,sizeof(T)*B.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }
        size_t *coffsets=C.pblock_offsets;
        size_t *cstrides=C.pstridesbuffer;
        T* cdata=C.pdata;
        if(ongpu)
        {
            if (C.plocal_blocknumber > 0)
            {
                size_t length_extents_strides=sizeof(size_t)*2*C.plocal_blocknumber;
                coffsets=(size_t*) omp_target_alloc(length_extents_strides,devnum);
                omp_target_memcpy(coffsets,C.pblock_offsets,length_extents_strides,0,0,devnum,omp_get_initial_device());
                cstrides=(size_t*) omp_target_alloc(length_extents_strides,devnum);
                omp_target_memcpy(cstrides,C.pstridesbuffer,length_extents_strides,0,0,devnum,omp_get_initial_device());

                if(!C.pdpdata_is_devptr)
                {
                    cdata=(T*) omp_target_alloc(sizeof(T)*C.pdatalength,devnum);
                    omp_target_memcpy(cdata,C.pdata,sizeof(T)*C.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }
        struct BlockMeta
        {
            size_t block_row;
            size_t block_col;
            size_t rows;
            size_t cols;
            size_t str0;
            size_t str1;
            size_t length;
        };

        for (size_t k = 0; k < grid_k; k++)
        {
            BlockMeta A_meta{0,0,0,0,0};
            int root_col = k % Pc;
            T* A_ptr = A_buf;
            if (my_col == root_col)
            {
                size_t A_lin = my_row * grid_k + k;
                auto it = A.pglobal_to_local_index.find(A_lin);
                if (it != A.pglobal_to_local_index.end())
                {
                    size_t idx = it->second;
                    A_meta.block_row = A.pblock_coords[2 * idx];
                    A_meta.block_col = A.pblock_coords[2 * idx + 1];
                    A_meta.rows = A.pblocks[idx].dpextents[0];
                    A_meta.cols = A.pblocks[idx].dpextents[1];
                    A_meta.str0 = A.pblocks[idx].dpstrides[0];
                    A_meta.str1 = A.pblocks[idx].dpstrides[1];
                    A_meta.length = A_meta.rows * A_meta.cols;
                    size_t offset = A.pblocks[idx].dpdata - A.pdata;
                    A_ptr = adata + offset;

                }
            }
            MPI_Bcast(&A_meta, sizeof(BlockMeta), MPI_BYTE, root_col, row_comm);
            MPI_Bcast(A_ptr, A_meta.length, mpi_get_type<T>(), root_col, row_comm);
            int root_row = k % Pr;
            BlockMeta B_meta{0,0,0,0,0};

            T* B_ptr = B_buf;
            if (my_row == root_row)
            {
                size_t B_lin = k * grid_c + my_col;

                auto it = B.pglobal_to_local_index.find(B_lin);
                if (it != B.pglobal_to_local_index.end())
                {
                    size_t idx = it->second;
                    B_meta.block_row = B.pblock_coords[2 * idx];
                    B_meta.block_col = B.pblock_coords[2 * idx + 1];
                    B_meta.rows = B.pblocks[idx].dpextents[0];
                    B_meta.cols = B.pblocks[idx].dpextents[1];
                    B_meta.str0 = B.pblocks[idx].dpstrides[0];
                    B_meta.str1 = B.pblocks[idx].dpstrides[1];
                    B_meta.length = B_meta.rows * B_meta.cols;
                    size_t offset = B.pblocks[idx].dpdata - B.pdata;
                    B_ptr = bdata + offset;
                }
            }
            MPI_Bcast(&B_meta, sizeof(BlockMeta), MPI_BYTE, root_row, col_comm);
            MPI_Bcast(B_ptr, B_meta.length, mpi_get_type<T>(), root_row, col_comm);
            const size_t A_block_rows=A_meta.rows;
            const size_t A_block_cols=A_meta.cols;
            const size_t B_block_cols=B_meta.cols;
            const size_t A_block_str0=A_meta.str0;
            const size_t A_block_str1=A_meta.str1;
            const size_t B_block_str0=B_meta.str0;
            const size_t B_block_str1=B_meta.str1;


            if (C.plocal_blocknumber > 0 && A_meta.length > 0 && B_meta.length > 0)
            {
                if(ongpu)
                {
                    const size_t Cstr0=cstrides[0];
                    const size_t Cstr1=cstrides[1];
                    T* C_ptr=cdata+coffsets[0];
                    #pragma omp target teams distribute parallel for collapse(2)device(devnum) is_device_ptr(cdata,A_ptr,B_ptr)
                    for (size_t ir = 0; ir < A_block_rows; ++ir)
                    {
                        for (size_t j = 0; j < B_block_cols; ++j)
                        {
                            T sum =T(0);
                            #pragma omp simd reduction(+:sum)
                            for (size_t p = 0; p < A_block_cols; ++p)
                            {
                                sum += A_ptr[ir * A_block_str0 + p * A_block_str1] *B_ptr[p  * B_block_str0 + j * B_block_str1];
                            }
                            C_ptr[ir*Cstr0+j*Cstr1]+= sum;
                        }
                    }
                }
                else
                {

                    T* C_ptr=cdata+coffsets[0];
                    const size_t Cstr0=cstrides[0];
                    const size_t Cstr1=cstrides[1];
                    #pragma omp parallel for collapse(2)
                    for (size_t ir = 0; ir < A_block_rows; ++ir)
                    {
                        for (size_t j = 0; j < B_block_cols; ++j)
                        {

                            T sum =T(0);
                            #pragma omp simd reduction(+:sum)
                            for (size_t k = 0; k < A_block_cols; ++k)
                            {
                                sum += A_ptr[ir*A_block_str0+k*A_block_str1] *B_ptr[k*B_block_str0+j*B_block_str1];
                            }
                            C_ptr[ir*Cstr0+j*Cstr1]+= sum;
                        }
                    }
                }
            }
        }

        if(ongpu)
        {
            if (A.plocal_blocknumber > 0)
            {
                if(!A.pdpdata_is_devptr)
                    omp_target_free(adata,devnum);
            }
            if (B.plocal_blocknumber > 0)
            {
                if(!B.pdpdata_is_devptr)
                    omp_target_free(bdata,devnum);
            }

            if (C.plocal_blocknumber>0)
            {
                if(!C.pdpdata_is_devptr)
                {
                    omp_target_memcpy(C.pdata,cdata,sizeof(T)*C.pdatalength,0,0,omp_get_initial_device(),devnum);
                    omp_target_free(cdata,devnum);
                }
                omp_target_free(cstrides,devnum);
                omp_target_free(coffsets,devnum);
            }
        }

        if(max_A>0)
            DataBlock_MPI_Functions<T>::free_helper2(memmap,ongpu, devnum,max_A,A_buf);
        if(max_B>0)
            DataBlock_MPI_Functions<T>::free_helper2(memmap, ongpu,devnum,max_B,B_buf);

        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);

        return true;
    }

    else
    {
        MPI_Comm comm;
        int Pr,Pc;
        int coords[2];
        if(minnumber==0)
        {
            int color = (C.plocal_blocknumber > 0) ? 1 : MPI_UNDEFINED;

            MPI_Comm comm_temp;
            MPI_Comm_split(comma, color, rank, &comm_temp);

            if (comm_temp == MPI_COMM_NULL)
                return true;

            int active_size;
            MPI_Comm_size(comm_temp, &active_size);

            int dims[2] = {0,0};
            MPI_Dims_create(active_size, 2, dims);
            Pr=dims[0];
            Pc=dims[1];
            int periods[2] = {0,0};
            MPI_Cart_create(comm_temp, 2, dims, periods, 1, &comm);
            int new_rank;
            MPI_Comm_rank(comm, &new_rank);
            MPI_Cart_coords(comm, new_rank, 2, coords);
        }
        else
        {
            comm=comma;
            Pr = C.pctx->dims[0];
            Pc = C.pctx->dims[1];
            MPI_Cart_coords(comm, rank, 2, coords);
        }

        int my_row = coords[0];
        int my_col = coords[1];


        const size_t br = A.pblock_extents[0];
        const size_t bk = A.pblock_extents[1];
        const size_t bc = B.pblock_extents[1];
        const size_t M = A.pglobal_extents[0];
        const size_t N = B.pglobal_extents[1];
        const size_t Ktot = A.pglobal_extents[1];
        const size_t grid_r = (M + br - 1) / br;
        const size_t grid_c = (N + bc - 1) / bc;
        const size_t grid_k = (Ktot + bk - 1) / bk;

        if (Pr > grid_r || Pc > grid_c)
            return false;

        MPI_Comm row_comm, col_comm;

        MPI_Comm_split(comm, my_row, my_col, &row_comm);
        MPI_Comm_split(comm, my_col, my_row, &col_comm);
        const size_t max_A = br * bk;
        const size_t max_B = bk * bc;
        bool ongpu=policy.should_use_gpu(A,B,C,Math_Functions_Policy::default_cubic_treshold);
        bool memmap=policy.memmapped_files;
        int devnum=policy.devicenum;
        if(A.pdpdata_is_devptr&& A.pdevptr_devicenum!=devnum) return false;
        if(B.pdpdata_is_devptr&& B.pdevptr_devicenum!=devnum) return false;
        if(C.pdpdata_is_devptr&& C.pdevptr_devicenum!=devnum) return false;
        if(A.pdevptr_devicenum!=B.pdevptr_devicenum ||A.pdevptr_devicenum!=C.pdevptr_devicenum) return false;


        struct PanelPair
        {
            size_t a_idx;
            size_t b_idx;
            size_t c_idx;
        };

        size_t num_A_panels = 0;
        size_t num_B_panels = 0;

        size_t* Ci_list = new size_t[C.plocal_blocknumber];
        size_t* Cj_list = new size_t[C.plocal_blocknumber];


        bool* mark = new bool[grid_r > grid_c ? grid_r : grid_c];

        #pragma omp parallel for simd if(parallel:grid_r>30)
        for (size_t i = 0; i < grid_r; i++) mark[i] = false;


        for (size_t i = 0; i < C.plocal_blocknumber; i++)
        {
            size_t Ci = C.pblock_coords[2*i] / br;

            if (!mark[Ci])
            {
                mark[Ci] = true;
                Ci_list[num_A_panels++] = Ci;
            }
        }

        #pragma omp parallel for simd if(parallel: grid_c>30)
        for (size_t j = 0; j < grid_c; j++) mark[j] = false;


        for (size_t i = 0; i < C.plocal_blocknumber; i++)
        {
            size_t Cj = C.pblock_coords[2*i+1] / bc;

            if (!mark[Cj])
            {
                mark[Cj] = true;
                Cj_list[num_B_panels++] = Cj;
            }
        }

        delete[] mark;



        T* A_buf;
        T* B_buf;
        if(max_A>0)
            DataBlock_MPI_Functions<T>::alloc_helper2(memmap,ongpu,devnum,max_A*num_A_panels,A_buf);
        if(max_B>0)
            DataBlock_MPI_Functions<T>::alloc_helper2(memmap,ongpu,devnum,max_B*num_B_panels,B_buf);

        T* adata=A.pdata;
        if(ongpu)
        {
            if(A.plocal_blocknumber > 0)
            {
                if(!A.pdpdata_is_devptr)
                {
                    adata=(T*) omp_target_alloc(sizeof(T)*A.pdatalength,devnum);
                    omp_target_memcpy(adata,A.pdata,sizeof(T)*A.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }
        T* bdata=B.pdata;
        if(ongpu)
        {
            if (B.plocal_blocknumber > 0)
            {
                if(!B.pdpdata_is_devptr)
                {
                    bdata=(T*) omp_target_alloc(sizeof(T)*B.pdatalength,devnum);
                    omp_target_memcpy(bdata,B.pdata,sizeof(T)*B.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }
        size_t *coffsets=C.pblock_offsets;
        size_t *cstrides=C.pstridesbuffer;
        size_t *cblockcoords=C.pblock_coords;
        T* cdata=C.pdata;
        if(ongpu)
        {
            if (C.plocal_blocknumber > 0)
            {
                size_t length_extents_strides=sizeof(size_t)*2*C.plocal_blocknumber;
                coffsets=(size_t*) omp_target_alloc(length_extents_strides,devnum);
                omp_target_memcpy(coffsets,C.pblock_offsets,length_extents_strides,0,0,devnum,omp_get_initial_device());
                cstrides=(size_t*) omp_target_alloc(length_extents_strides,devnum);
                omp_target_memcpy(cstrides,C.pstridesbuffer,length_extents_strides,0,0,devnum,omp_get_initial_device());
                cblockcoords=(size_t*) omp_target_alloc(length_extents_strides,devnum);
                omp_target_memcpy(cblockcoords,C.pblock_coords,length_extents_strides,0,0,devnum,omp_get_initial_device());
                if(!C.pdpdata_is_devptr)
                {
                    cdata=(T*) omp_target_alloc(sizeof(T)*C.pdatalength,devnum);
                    omp_target_memcpy(cdata,C.pdata,sizeof(T)*C.pdatalength,0,0,devnum,omp_get_initial_device());
                }
            }
        }

        struct BlockMeta
        {
            size_t block_row;
            size_t block_col;
            size_t rows;
            size_t cols;
            size_t str0;
            size_t str1;
            size_t length;
        };

        BlockMeta* A_meta_arr = new BlockMeta[num_A_panels];
        BlockMeta* B_meta_arr = new BlockMeta[num_B_panels];
        T** A_panel_ptrs = new T*[num_A_panels];
        T** B_panel_ptrs = new T*[num_B_panels];

        for (size_t k = 0; k < grid_k; k++)
        {
            for (size_t p = 0; p < num_A_panels; p++)
            {
                const size_t bi = Ci_list[p];

                BlockMeta& A_meta = A_meta_arr[p];
                A_meta = {0,0,0,0,0,0,0};

                const int root_col = k % Pc;

                T* root_ptr = nullptr;
                T* recv_ptr = A_buf + p * max_A;

                if (my_col == root_col)
                {
                    const size_t A_lin = bi * grid_k + k;

                    auto it = A.pglobal_to_local_index.find(A_lin);
                    if (it != A.pglobal_to_local_index.end())
                    {
                        size_t idx = it->second;

                        A_meta.block_row = A.pblock_coords[2 * idx];
                        A_meta.block_col = A.pblock_coords[2 * idx + 1];
                        A_meta.rows = A.pblocks[idx].dpextents[0];
                        A_meta.cols = A.pblocks[idx].dpextents[1];
                        A_meta.str0 = A.pblocks[idx].dpstrides[0];
                        A_meta.str1 = A.pblocks[idx].dpstrides[1];
                        A_meta.length = A_meta.rows * A_meta.cols;

                        const size_t offset = A.pblocks[idx].dpdata - A.pdata;
                        root_ptr = adata + offset;
                    }
                }

                MPI_Bcast(&A_meta, sizeof(BlockMeta), MPI_BYTE, root_col, row_comm);


                if (my_col == root_col)
                    A_panel_ptrs[p] = root_ptr;
                else
                    A_panel_ptrs[p] = recv_ptr;

                MPI_Bcast(A_panel_ptrs[p], A_meta.length, mpi_get_type<T>(), root_col, row_comm);
            }

            for (size_t p = 0; p < num_B_panels; p++)
            {
                const size_t bj = Cj_list[p];

                BlockMeta& B_meta = B_meta_arr[p];
                B_meta = {0,0,0,0,0,0,0};

                const int root_row = k % Pr;

                T* root_ptr = nullptr;
                T* recv_ptr = B_buf + p * max_B;

                if (my_row == root_row)
                {
                    const size_t B_lin = k * grid_c + bj;

                    auto it = B.pglobal_to_local_index.find(B_lin);
                    if (it != B.pglobal_to_local_index.end())
                    {
                        size_t idx = it->second;

                        B_meta.block_row = B.pblock_coords[2 * idx];
                        B_meta.block_col = B.pblock_coords[2 * idx + 1];
                        B_meta.rows = B.pblocks[idx].dpextents[0];
                        B_meta.cols = B.pblocks[idx].dpextents[1];
                        B_meta.str0 = B.pblocks[idx].dpstrides[0];
                        B_meta.str1 = B.pblocks[idx].dpstrides[1];
                        B_meta.length = B_meta.rows * B_meta.cols;

                        const size_t offset = B.pblocks[idx].dpdata - B.pdata;
                        root_ptr = bdata + offset;
                    }
                }

                MPI_Bcast(&B_meta, sizeof(BlockMeta), MPI_BYTE, root_row, col_comm);

                if (my_row == root_row)
                    B_panel_ptrs[p] = root_ptr;
                else
                    B_panel_ptrs[p] = recv_ptr;

                MPI_Bcast(B_panel_ptrs[p],B_meta.length,mpi_get_type<T>(),root_row, col_comm);
            }

            if(ongpu)
            {
                size_t cblocknumber=C.plocal_blocknumber;
                #pragma omp target data map(to: A_meta_arr[0:num_A_panels]) map(to: B_meta_arr[0:num_B_panels])device(devnum)
                {
                    #pragma omp target teams distribute collapse(2) device(devnum) is_device_ptr(cdata,coffsets,cblockcoords,A_panel_ptrs,B_panel_ptrs)
                    for (size_t cpi = 0; cpi < num_A_panels; cpi++)
                    {
                        for (size_t cpj = 0; cpj < num_B_panels; cpj++)
                        {
                            const T* A_ptr = A_panel_ptrs[cpi];
                            const T* B_ptr = B_panel_ptrs[cpj];

                            const BlockMeta& A_meta = A_meta_arr[cpi];
                            const BlockMeta& B_meta = B_meta_arr[cpj];

                            #pragma omp parallel for collapse(3)
                            for (size_t i = 0; i <cblocknumber; i++)
                            {
                                for (size_t r = 0; r < A_meta.rows; r++)
                                {
                                    for (size_t c = 0; c < B_meta.cols; c++)
                                    {
                                        if (cblockcoords[2*i] != A_meta.block_row ||
                                                cblockcoords[2*i+1] != B_meta.block_col)
                                            continue;

                                        T* C_ptr = cdata + coffsets[i];

                                        T sum = 0;
                                        #pragma omp simd reduction(+:sum)
                                        for (size_t k = 0; k < A_meta.cols; k++)
                                        {
                                            sum += A_ptr[r*A_meta.str0 + k*A_meta.str1] *
                                                   B_ptr[k*B_meta.str0 + c*B_meta.str1];
                                        }

                                        C_ptr[r*cstrides[2*i] + c*cstrides[2*i+1]] += sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                #pragma omp parallel for collapse(2)
                for (size_t cpi = 0; cpi < num_A_panels; cpi++)
                {
                    for (size_t cpj = 0; cpj < num_B_panels; cpj++)
                    {
                        const T* A_ptr = A_panel_ptrs[cpi];
                        const T* B_ptr = B_panel_ptrs[cpj];

                        const BlockMeta& A_meta = A_meta_arr[cpi];
                        const BlockMeta& B_meta = B_meta_arr[cpj];

                        for (size_t i = 0; i < C.plocal_blocknumber; i++)
                        {
                            for (size_t r = 0; r <  A_meta.rows; r++)
                            {
                                for (size_t c = 0; c < B_meta.cols; c++)
                                {
                                    if (cblockcoords[2*i] != A_meta.block_row ||
                                            cblockcoords[2*i+1] != B_meta.block_col)
                                        continue;

                                    T* C_ptr = cdata + coffsets[i];

                                    T sum = 0;
                                    #pragma omp simd reduction(+:sum)
                                    for (size_t k = 0; k < A_meta.cols; k++)
                                    {
                                        sum += A_ptr[r*A_meta.str0 + k*A_meta.str1] *
                                               B_ptr[k*B_meta.str0 + c*B_meta.str1];
                                    }

                                    C_ptr[r*cstrides[2*i] + c*cstrides[2*i+1]] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }

        delete[] A_panel_ptrs;
        delete[] B_panel_ptrs;
        delete[] A_meta_arr ;
        delete[] B_meta_arr;
        delete[] Ci_list ;
        delete[] Cj_list ;

        if(ongpu)
        {
            if (A.plocal_blocknumber > 0)
            {
                if(!A.pdpdata_is_devptr)
                    omp_target_free(adata,devnum);
            }
            if (B.plocal_blocknumber > 0)
            {
                if(!B.pdpdata_is_devptr)
                    omp_target_free(bdata,devnum);
            }

            if (C.plocal_blocknumber>0)
            {
                if(!C.pdpdata_is_devptr)
                {
                    omp_target_memcpy(C.pdata,cdata,sizeof(T)*C.pdatalength,0,0,omp_get_initial_device(),devnum);
                    omp_target_free(cdata,devnum);
                }
                omp_target_free(cstrides,devnum);
                omp_target_free(coffsets,devnum);
                omp_target_free(cblockcoords,devnum);
            }
        }
        if(max_A>0)
            DataBlock_MPI_Functions<T>::free_helper2(memmap,ongpu, devnum,max_A*num_A_panels,A_buf);
        if(max_B>0)
            DataBlock_MPI_Functions<T>::free_helper2(memmap, ongpu,devnum,max_B*num_B_panels,B_buf);

        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);

        return true;

    }
}




template<typename T>
inline bool Math_Functions_MPI<T>::Matrix_Vector_multiply_Distributed(
    const DistributedDataBlock<T>& A,
    const DistributedDataBlock<T>& x,
    DistributedDataBlock<T>& y,
    const Math_MPI_Functions_Policy*pol)
{
    const Math_MPI_Functions_Policy policy = (pol != nullptr) ? *pol : get_default_policy();
    if (A.pctx->comm == MPI_COMM_NULL)
        return false;

    int rank, size;
    MPI_Comm_rank(A.pctx->comm, &rank);
    MPI_Comm_size(A.pctx->comm, &size);

    const size_t M = A.pglobal_extents[0];
    const size_t K = A.pglobal_extents[1];

    const size_t br = A.pblock_extents[0];
    const size_t bc = A.pblock_extents[1];
    const size_t bs = y.pblock_extents[0];

    const size_t grid_c = (K + bc - 1) / bc;
    const size_t grid_r = (M + bs - 1) / bs;
    bool ongpu=policy.should_use_gpu(A,x,y,Math_Functions_Policy::default_cubic_treshold);
    bool memmap=policy.memmapped_files;

    int devnum=policy.devicenum;

    if(A.pdpdata_is_devptr&& A.pdevptr_devicenum!=devnum)
        return false;
    if(x.pdpdata_is_devptr&& x.pdevptr_devicenum!=devnum)
        return false;
    if(y.pdpdata_is_devptr&& y.pdevptr_devicenum!=devnum)
        return false;

    if(ongpu)
    {
        if(y.pdevptr_devicenum!=x.pdevptr_devicenum ||A.pdevptr_devicenum!=y.pdevptr_devicenum)
            return false;
    }

    T* x_global=nullptr;
    if(K>0)
        DataBlock_MPI_Functions<T>::alloc_helper2(memmap,ongpu,devnum,K,x_global);

    if (ongpu)
    {
        #pragma omp target teams distribute parallel for simd device(devnum)
        for (size_t i=0; i<K; i++)
            x_global[i]=0;
    }
    else
    {
        #pragma omp parallel for simd
        for (size_t i=0; i<K; i++)
            x_global[i]=0;
    }



    if(ongpu)
    {
        for (size_t i = 0; i < x.plocal_blocknumber; i++)
        {
            size_t b = x.pblock_linear_idx[i];
            size_t start = b * bc;

            size_t diff = K - start;
            size_t len  = (bc < diff) ? bc : diff;

            size_t off = x.pblock_offsets[i];

            if(x.pdpdata_is_devptr)
            {
                omp_target_memcpy_async(x_global,x.pdata,len*sizeof(T),sizeof(T)*start,sizeof(T)*off,devnum,x.pdevptr_devicenum,0);
            }
            else
            {
                omp_target_memcpy_async(x_global,x.pdata,len*sizeof(T),sizeof(T)*start,sizeof(T)*off,devnum,omp_get_initial_device(),0);
            }
        }
        #pragma omp taskwait

    }
    else
    {
        #pragma omp parallel for if(parallel:x.plocal_blocknumber>30)
        for (size_t i = 0; i < x.plocal_blocknumber; i++)
        {
            size_t b = x.pblock_linear_idx[i];
            size_t start = b * bc;

            size_t diff = K - start;
            size_t len  = (bc < diff) ? bc : diff;

            size_t off = x.pblock_offsets[i];

            const T* src = x.pdata + off;
            T* dst = x_global + start;
            memcpy(dst, src, len * sizeof(T));
        }
    }

    MPI_Allreduce(
        MPI_IN_PLACE,
        x_global,
        K,
        mpi_get_type<T>(),
        MPI_SUM,
        A.pctx->comm);

    T* y_full=nullptr, *A_ptr=nullptr;
    size_t* Aext=nullptr,*Ablockoff=nullptr,*Ablocklinindex=nullptr;

    if(M>0)
        DataBlock_MPI_Functions<T>::alloc_helper2(memmap,ongpu,devnum,M,y_full);


    bool rowm=A.pglobal_rowmajor;
    if(A.pdatalength>0 &&A.plocal_blocknumber>0)
    {
        if(ongpu)
        {
            if(!A.pdpdata_is_devptr)
            {
                A_ptr=(T*) omp_target_alloc(sizeof(T)*A.pdatalength,devnum);
                omp_target_memcpy_async(A_ptr,A.pdata,sizeof(T)*A.pdatalength,0,0,devnum,omp_get_initial_device(),0);
            }
            else
                A_ptr=A.pdata;

            Ablockoff=(size_t*) omp_target_alloc(sizeof(size_t)*A.plocal_blocknumber,devnum);
            omp_target_memcpy_async(Ablockoff,A.pblock_offsets, sizeof(size_t)*A.plocal_blocknumber,0,0,devnum,omp_get_initial_device(),0);

            Aext=(size_t*) omp_target_alloc(sizeof(T)*A.plocal_blocknumber*2,devnum );
            omp_target_memcpy_async(Aext, A.pextentsbuffer, sizeof(size_t)*A.plocal_blocknumber*2,0,0,devnum,omp_get_initial_device(),0);

            Ablocklinindex=(size_t*) omp_target_alloc(sizeof(size_t)*A.plocal_blocknumber,devnum);
            omp_target_memcpy_async(Ablocklinindex,A.pblock_linear_idx,sizeof(size_t*)*A.plocal_blocknumber,0,0,devnum, omp_get_initial_device(),0);


            const size_t num=A.plocal_blocknumber;
            #pragma omp taskwait

            #pragma omp target teams distribute parallel for \
            is_device_ptr(Ablocklinindex,Aext,Ablockoff,A_ptr,y_full) \
            device(devnum)
            for (size_t global_row = 0; global_row < M; global_row++)
            {
                T total = T(0);

                for (size_t bi_local = 0; bi_local < num; bi_local++)
                {
                    const size_t b = Ablocklinindex[bi_local];

                    const size_t bi = b / grid_c;
                    const size_t bj = b % grid_c;

                    const size_t row0 = bi * br;
                    const size_t col0 = bj * bc;

                    const size_t rows = Aext[bi_local * 2 + 0];

                    if (global_row >= row0 && global_row < row0 + rows)
                    {
                        const size_t r = global_row - row0;

                        const size_t a_off = Ablockoff[bi_local];
                        const size_t cols  = Aext[bi_local * 2 + 1];

                        T sum = T(0);

                        if (rowm)
                        {
                            const size_t a_row_off = a_off + r * cols;

                            #pragma omp simd reduction(+:sum)
                            for (size_t c = 0; c < cols; c++)
                            {
                                sum += A_ptr[a_row_off + c] * x_global[col0 + c];
                            }
                        }
                        else
                        {
                            #pragma omp simd reduction(+:sum)
                            for (size_t c = 0; c < cols; c++)
                            {
                                const size_t a_idx = a_off + c * rows + r;
                                sum += A_ptr[a_idx] * x_global[col0 + c];
                            }
                        }

                        total += sum;
                    }
                }
                y_full[global_row] = total;
            }
        }
        else
        {
            Aext= A.pextentsbuffer;
            Ablockoff=A.pblock_offsets;
            Ablocklinindex=A.pblock_linear_idx;
            A_ptr=A.pdata;
            const size_t num=A.plocal_blocknumber;
            #pragma omp parallel for
            for (size_t global_row = 0; global_row < M; global_row++)
            {
                T total = T(0);

                for (size_t bi_local = 0; bi_local < num; bi_local++)
                {
                    const size_t b = Ablocklinindex[bi_local];

                    const size_t bi = b / grid_c;
                    const size_t bj = b % grid_c;

                    const size_t row0 = bi * br;
                    const size_t col0 = bj * bc;

                    const size_t rows = Aext[bi_local * 2 + 0];

                    if (global_row >= row0 && global_row < row0 + rows)
                    {
                        const size_t r = global_row - row0;

                        const size_t a_off = Ablockoff[bi_local];
                        const  size_t cols  = Aext[bi_local * 2 + 1];

                        T sum = T(0);

                        if (rowm)
                        {
                            const size_t a_row_off = a_off + r * cols;

                            #pragma omp simd reduction(+:sum)
                            for (size_t c = 0; c < cols; c++)
                            {
                                sum += A_ptr[a_row_off + c] * x_global[col0 + c];
                            }
                        }
                        else
                        {
                            #pragma omp simd reduction(+:sum)
                            for (size_t c = 0; c < cols; c++)
                            {
                                const size_t a_idx = a_off + c * rows + r;
                                sum += A_ptr[a_idx] * x_global[col0 + c];
                            }
                        }

                        total += sum;
                    }
                }
                y_full[global_row] = total;
            }
        }
    }

    if(K>0)
        DataBlock_MPI_Functions<double>::free_helper2(memmap,ongpu, devnum,  K,x_global);

    int* recvcounts = new int[size];

    #pragma omp parallel for simd if(parallel:size>30)
    for (size_t i=0; i<size; i++)
        recvcounts[i]=0;

    int ndims;
    MPI_Cartdim_get(y.pctx->comm, &ndims);
    size_t gridrank=(size_t) ndims;
    size_t *gridcoords=new size_t [gridrank];
    int* tempcoords=new int[gridrank];

    for (size_t b = 0; b < grid_r; b++)
    {
        size_t diff= M - b * bs;
        size_t len = bs<diff? bs:diff;

        size_t bcoords[1] = { b };

        y.ppolicy->create_coords( bcoords,gridcoords,  y.pglobalrank);
        int owner = y.ppolicy->owner(gridcoords, *y.pctx, tempcoords);

        recvcounts[owner] += (int)len;
    }
    delete[] gridcoords;
    delete[] tempcoords;

    T* y_local=nullptr;

    if(recvcounts[rank]>0)
        DataBlock_MPI_Functions<T>::alloc_helper2(memmap,ongpu,devnum,recvcounts[rank],y_local);


    MPI_Reduce_scatter(
        y_full,
        y_local,
        recvcounts,
        mpi_get_type<T>(),
        MPI_SUM,
        y.pctx->comm);

    if(M>0)
        DataBlock_MPI_Functions<double>::free_helper2(memmap,ongpu, devnum,  M,y_full);

    if(ongpu)
    {
        if(!A.pdpdata_is_devptr)
            omp_target_free(A_ptr,devnum);

        omp_target_free(Ablockoff,devnum);
        omp_target_free(Aext,devnum);
        omp_target_free(Ablocklinindex,devnum);
    }


    size_t offset = 0;
    #pragma omp parallel for simd reduction(+:offset)if(parallel:rank>30)
    for (int i = 0; i < rank; i++)
        offset += recvcounts[i];

    if(ongpu)
    {
        for (size_t i = 0; i < y.plocal_blocknumber; i++)
        {
            size_t b = y.pblock_linear_idx[i];

            size_t start = b * bs;
            size_t diff  = M - start;
            size_t len   = (bs < diff) ? bs : diff;

            size_t global_offset = start;

            size_t local_offset = global_offset - offset;

            size_t dst = y.pblock_offsets[i];
            if(!y.pdpdata_is_devptr)
                omp_target_memcpy_async( y.pdata,y_local,len * sizeof(T),dst*sizeof(T),local_offset*sizeof(T), omp_get_initial_device(),devnum,0);
            else
                omp_target_memcpy_async(    y.pdata,    y_local,len * sizeof(T   ),dst*sizeof(T),local_offset*sizeof(T),y.pdevptr_devicenum,devnum,0);

        }
        #pragma omp taskwait

    }

    else
    {
        for (size_t i = 0; i < y.plocal_blocknumber; i++)
        {
            size_t b = y.pblock_linear_idx[i];

            size_t start = b * bs;
            size_t diff  = M - start;
            size_t len   = (bs < diff) ? bs : diff;

            size_t global_offset = start;

            size_t local_offset = global_offset - offset;

            size_t dst = y.pblock_offsets[i];
            memcpy(    y.pdata + dst,    y_local + local_offset,len * sizeof(T   ));

        }
    }
    if(recvcounts[rank]>0)
        DataBlock_MPI_Functions<T>::free_helper2(memmap,ongpu,devnum,recvcounts[rank],y_local);

    delete[] recvcounts;

    return true;
}

template <typename T>
void Math_Functions_MPI<T>::strassen_multiply( const DataBlock<T> & A,const  DataBlock<T> & B, DataBlock<T> & C,const Math_MPI_RecursiveMultiplication_Policy *pol)
{
    const Math_MPI_RecursiveMultiplication_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    bool ongpu=policy.should_use_gpu(A,B,C,Math_Functions_Policy::default_cubic_treshold,7);
    bool separate_device_memory=false;
    if(ongpu)
    {
#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif
    }

    if(separate_device_memory)
    {
        typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadA(A, policy.devicenum, false);
        typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadB(B,  policy.devicenum, false);
        typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadC(C,  policy.devicenum, true, policy.update_host);
        DataBlock<T> tA=A,tB=B,tC=C;

        if(!tA.dpdata_is_devptr)
            tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);
        if(!tB.dpdata_is_devptr)
            tB.dpdata=(T*) omp_get_mapped_ptr(B.dpdata,policy.devicenum);
        if(!tC.dpdata_is_devptr)
            tC.dpdata=(T*) omp_get_mapped_ptr(C.dpdata,policy.devicenum);

        tA.dpdata_is_devptr=true;
        tB.dpdata_is_devptr=true;
        tC.dpdata_is_devptr=true;
        tA.devptr_devicenum=policy.devicenum;
        tB.devptr_devicenum=policy.devicenum;
        tC.devptr_devicenum=policy.devicenum;

        strassen_multiply_h(tA,tB,tC,ongpu, separate_device_memory,policy);
    }
    else
    {
        strassen_multiply_h(A,B,C,ongpu, false,policy);
    }

}

template <typename T>
void Math_Functions_MPI<T>::strassen_multiply_h(const DataBlock<T> & A, const DataBlock<T> & B, DataBlock<T> & C,bool ongpu, bool separate_device_memory, const Math_MPI_RecursiveMultiplication_Policy &policy)
{


    // Dimensions of input matrices
    size_t n = A.dpextents[0]; // Rows in A
    size_t m = A.dpextents[1]; // Columns in A and rows in B
    size_t p = A.dpextents[1]; // Columns in B


    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || !policy.should_use_recursion(n*p))
    {
        if(ongpu)
        {
            GPU_Math_Functions<T>::matrix_multiply_dot_g(   A,B,  C,policy.devicenum,false);
            return;
        }
        else
        {
            switch (policy.mode)
            {
            case Math_Functions_Policy::GPU_ONLY:
            {
                GPU_Math_Functions<T>::matrix_multiply_dot_g(A,B,C,policy.devicenum,true);
                return;
            }
            case Math_Functions_Policy::AUTO:
            {
                if(policy.should_use_gpu(A,B,C,Math_Functions_Policy::default_cubic_treshold,1))
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(A,B,C,policy.devicenum,true);
                else
                    In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w( A,B,C);
                return;
            }
            default:
            {
                In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w( A,B,  C);
                return;
            }
            }
        }
    }


    size_t half_n = n / 2;
    size_t half_m = m / 2;
    size_t half_p = p / 2;

// Submatrices of A

    size_t psext1[2],a11str[2],psext2[2],a12str[2],psext3[2],a21str[2],psext4[2],a22str[2],
           psext5[2],b11str[2],psext6[2],b12str[2],psext7[2],b21str[2],psext8[2],b22str[2];




// Temporary storage for intermediate results
    const size_t s=half_n*half_p,
                 s2=half_n*half_m,
                 s3=half_m*half_p;

    size_t ext1[2]= {half_n, half_p};
    size_t str1[2]= {half_p, 1};


    size_t ext2[2]= {half_n, half_m};
    size_t str2[2]= {half_m, 1};


    size_t ext3[2]=  {half_m, half_p};
    size_t str3[2]= {half_p, 1};




    T* Ard1,*Ard2,*Ard3,*Ard4,*Ard5,*Brd1,*Brd2,*Brd3,*Brd4,*Brd5,*M1d,*M2d,*M3d,*M4d,*M5d,*M6d,*M7d;

    if(separate_device_memory)
    {
        Ard1=(T*)omp_target_alloc(sizeof(T)*s2,policy.devicenum);
        Ard2=(T*)omp_target_alloc(sizeof(T)*s2,policy.devicenum);
        Ard3=(T*)omp_target_alloc(sizeof(T)*s2,policy.devicenum);
        Ard4=(T*)omp_target_alloc(sizeof(T)*s2,policy.devicenum);
        Ard5=(T*)omp_target_alloc(sizeof(T)*s2,policy.devicenum);

        Brd1=(T*)omp_target_alloc(sizeof(T)*s3,policy.devicenum);
        Brd2=(T*)omp_target_alloc(sizeof(T)*s3,policy.devicenum);
        Brd3=(T*)omp_target_alloc(sizeof(T)*s3,policy.devicenum);
        Brd4=(T*)omp_target_alloc(sizeof(T)*s3,policy.devicenum);
        Brd5=(T*)omp_target_alloc(sizeof(T)*s3,policy.devicenum);

        M1d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M2d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M3d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M4d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M5d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M6d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
        M7d=(T*)omp_target_alloc(sizeof(T)*s,policy.devicenum);
    }
    else
    {
        if(policy.memmapped_files)
        {
            Ard1=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s2);
            Ard2=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s2);
            Ard3=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s2);
            Ard4=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s2);
            Ard5=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s2);

            Brd1=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s3);
            Brd2=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s3);
            Brd3=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s3);
            Brd4=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s3);
            Brd5=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s3);

            M1d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M2d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M3d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M4d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M5d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M6d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M7d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
        }
        else
        {
            Ard1=new T[s2];
            Ard2=new T[s2];
            Ard3=new T[s2];
            Ard4=new T[s2];
            Ard5=new T[s2];

            Brd1=new T[s3];
            Brd2=new T[s3];
            Brd3=new T[s3];
            Brd4=new T[s3];
            Brd5=new T[s3];

            M1d=new T[s];
            M2d=new T[s];
            M3d=new T[s];
            M4d=new T[s];
            M5d=new T[s];
            M6d=new T[s];
            M7d=new T[s];
        }
    }


    DataBlock<T>
    A_result1(Ard1,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
              A_result2(Ard2,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
              A_result3(Ard3,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
              A_result4(Ard4,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
              A_result5(Ard5,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),

              B_result1(Brd1,s2,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
              B_result2(Brd2,s2,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
              B_result3(Brd3,s2,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
              B_result4(Brd4,s2,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
              B_result5(Brd5,s2,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),

              M1(M1d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M2(M2d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M3(M3d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M4(M4d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M5(M5d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M6(M6d,s,true,2,ext1,str1,false,false,separate_device_memory),
              M7(M7d,s,true,2,ext1,str1,false,false,separate_device_memory);


    DataBlock<T>  A11 = A.matrix_subspan(0, 0, half_n, half_m,psext1,a11str),
                  A12 = A.matrix_subspan(0, half_m, half_n, half_m,psext2,a12str),
                  A21 = A.matrix_subspan(half_n, 0, half_n, half_m,psext3,a21str),
                  A22 = A.matrix_subspan(half_n, half_m, half_n, half_m,psext4,a22str);

// Submatrices of B
    DataBlock<T>   B11 = B.matrix_subspan(0, 0, half_m, half_p,psext5,b11str),
                   B12 = B.matrix_subspan(0, half_p, half_m, half_p,psext6,b12str),
                   B21 = B.matrix_subspan(half_m, 0, half_m, half_p,psext7,b21str),
                   B22 = B.matrix_subspan(half_m, half_p, half_m, half_p,psext8,b22str);

    const size_t str20=str2[0];
    const size_t str21=str2[1];
    const size_t str30=str3[0];
    const size_t str31=str3[1];

    const size_t a11str0=a11str[0];
    const size_t a11str1=a11str[1];

    const size_t a12str0=a12str[0];
    const size_t a12str1=a12str[1];

    const size_t a21str0=a21str[0];
    const size_t a21str1=a21str[1];

    const size_t a22str0=a22str[0];
    const size_t a22str1=a22str[1];

    const size_t b11str0=b11str[0];
    const size_t b11str1=b11str[1];

    const size_t b12str0=b12str[0];
    const size_t b12str1=b12str[1];

    const size_t b21str0=b21str[0];
    const size_t b21str1=b21str[1];

    const size_t b22str0=b22str[0];
    const size_t b22str1=b22str[1];


    const T* A11d=A11.dpdata;
    const T* A12d=A12.dpdata;
    const T* A21d=A21.dpdata;
    const T* A22d=A22.dpdata;

    const T* B11d=B11.dpdata;
    const T* B12d=B12.dpdata;
    const T* B21d=B21.dpdata;
    const T* B22d=B22.dpdata;


    if (ongpu)
    {

        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum) is_device_ptr(Ard1,Ard2,Ard3,Ard4,Ard5,A11d,A12d,A21d,A22d)
        for (size_t i=0; i<half_n; i++)
        {
            for (size_t j=0; j<half_m; j++)
            {
                const T a11dd=A11d[i*a11str0+j*a11str1];
                const T a22dd=A22d[i*a22str0+j*a22str1];
                const T a21dd=A21d[i*a21str0+j*a21str1];
                const T a12dd=A12d[i*a12str0+j*a12str1];
                const size_t aindex=i*str20+j*str21;
                Ard1[aindex]=a11dd+a22dd;
                Ard2[aindex]=a21dd+a22dd;
                Ard3[aindex]=a11dd+a12dd;
                Ard4[aindex]=a21dd-a11dd;
                Ard5[aindex]=a12dd-a22dd;
            }
        }

        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum) is_device_ptr(Brd1,Brd2,Brd3,Brd4,Brd5,B11d,B12d,B21d,B22d)
        for (size_t i=0; i<half_m; i++)
        {
            for (size_t j=0; j<half_p; j++)
            {
                const T b11dd=B11d[i*b11str0+j*b11str1];
                const T b21dd=B21d[i*b21str0+j*b21str1];
                const T b12dd=B12d[i*b12str0+j*b12str1];
                const T b22dd=B22d[i*b22str0+j*b22str1];
                const size_t bindex=i*str30+j*str31;
                Brd1[bindex]=b11dd+b22dd;
                Brd2[bindex]=b12dd-b22dd;
                Brd3[bindex]=b21dd-b11dd;
                Brd4[bindex]=b11dd+b12dd;
                Brd5[bindex]=b21dd+b22dd;
            }
        }

    }
    else
    {
        #pragma omp parallel for simd collapse (2)
        for (size_t i=0; i<half_n; i++)
        {
            for (size_t j=0; j<half_m; j++)
            {
                const T a11dd=A11d[i*a11str0+j*a11str1];
                const T a22dd=A22d[i*a22str0+j*a22str1];
                const T a21dd=A21d[i*a21str0+j*a21str1];
                const T a12dd=A12d[i*a12str0+j*a12str1];
                const size_t aindex=i*str20+j*str21;
                Ard1[aindex]=a11dd+a22dd;
                Ard2[aindex]=a21dd+a22dd;
                Ard3[aindex]=a11dd+a12dd;
                Ard4[aindex]=a21dd-a11dd;
                Ard5[aindex]=a12dd-a22dd;
            }
        }

        #pragma omp parallel for simd collapse (2)
        for (size_t i=0; i<half_m; i++)
        {
            for (size_t j=0; j<half_p; j++)
            {
                const T b11dd=B11d[i*b11str0+j*b11str1];
                const T b21dd=B21d[i*b21str0+j*b21str1];
                const T b12dd=B12d[i*b12str0+j*b12str1];
                const T b22dd=B22d[i*b22str0+j*b22str1];
                const size_t bindex=i*str30+j*str31;
                Brd1[bindex]=b11dd+b22dd;
                Brd2[bindex]=b12dd-b22dd;
                Brd3[bindex]=b21dd-b11dd;
                Brd4[bindex]=b11dd+b12dd;
                Brd5[bindex]=b21dd+b22dd;
            }
        }
    }


    if (policy.should_use_mpi_for_recursion(7))
    {
        int myrank=0,childdest=0;

        MPI_Comm_rank(policy.comm, &myrank);
        childdest=myrank*7;

        int message=Math_MPI_RecursiveMultiplication_Policy::Strassen;


        MPI_Send(&message, 1, MPI_INT, childdest+1,0, policy.comm);
        size_t problemsize=s2;

        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+1,1, policy.comm);

        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(A_result1,childdest+1,2, policy.comm);

        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(B_result1,childdest+1,3, policy.comm);




        MPI_Send(&message, 1, MPI_INT, childdest+2, 0,  policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+2,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(A_result2,childdest+2,2, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(B11,childdest+2,3, policy.comm);



        MPI_Send(&message, 1, MPI_INT, childdest+3, 0,  policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+3,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(A11,childdest+3,2, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(B_result2,childdest+3,3, policy.comm);


        MPI_Send(&message, 1, MPI_INT, childdest+4, 0,  policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+4,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(A22,childdest+4,2, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(B_result3,childdest+4,3, policy.comm);


        MPI_Send(&message, 1, MPI_INT, childdest+5, 0,    policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+5,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(A_result3,childdest+5,2,   policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(B22,childdest+5,3,   policy.comm);


        MPI_Send(&message, 1, MPI_INT, childdest+6, 0,  policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+6,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(A_result4,childdest+6,2, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(B_result4,childdest+6,3, policy.comm);


        MPI_Send(&message, 1, MPI_INT, childdest+7, 0, policy.comm);
        problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+7,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(A_result5,childdest+7,2,   policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(B_result5,childdest+7,3,   policy.comm);

        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M1,childdest+1,4, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M2,childdest+2,4, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M3,childdest+3,4, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M4,childdest+4,4, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M5,childdest+5,4, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M6,childdest+6,4, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M7,childdest+7,4, policy.comm);

    }
    else
    {
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                {
                    strassen_multiply_h(A_result1, B_result1,   M1,ongpu, separate_device_memory, policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A_result2, B11,         M2,ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A11, B_result2,         M3,ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A22, B_result3,         M4,ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A_result3, B22,         M5,ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A_result4, B_result4,   M6,ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    strassen_multiply_h(A_result5, B_result5,   M7,ongpu,  separate_device_memory,policy);
                }
                #pragma omp taskwait
            }
        }
    }

    size_t ext11a[2],cstr11[2], ext12a[2],cstr12[2], ext13a[2],cstr21[2], ext14a[2],cstr22[2];

// Submatrices of C

    DataBlock<T>   C11 = C.matrix_subspan(0, 0, half_n, half_p,ext11a,cstr11),
                   C12 = C.matrix_subspan(0, half_p, half_n, half_p,ext12a,cstr12),
                   C21 = C.matrix_subspan(half_n, 0, half_n, half_p,ext13a,cstr21),
                   C22 = C.matrix_subspan(half_n, half_p, half_n, half_p,ext14a,cstr22);

    const size_t cstr110=cstr11[0];
    const size_t cstr111=cstr11[1];

    const size_t cstr120=cstr12[0];
    const size_t cstr121=cstr12[1];

    const size_t cstr210=cstr21[0];
    const size_t cstr211=cstr21[1];

    const size_t cstr220=cstr22[0];
    const size_t cstr221=cstr22[1];
    T* C11d=C11.dpdata;
    T* C12d=C12.dpdata;
    T* C21d=C21.dpdata;
    T* C22d=C22.dpdata;

    const size_t str10=str1[0];
    const size_t str11=str1[1];
    if(ongpu)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum) is_device_ptr(C11d,C12d,C21d,C22d,M1d,M2d,M3d,M4d,M5d,M6d)
        for (size_t i = 0; i < half_n; i++)
        {
            for (size_t j = 0; j < half_p; j++)
            {
                const size_t mindex=i*str10+j*str11;
                const T m1dd=M1d[mindex];
                const T m2dd=M2d[mindex];
                const T m3dd=M3d[mindex];
                const T m4dd=M4d[mindex];
                const T m5dd=M5d[mindex];
                const T m6dd=M6d[mindex];
                const T m7dd=M7d[mindex];

                const T helper1 = m1dd  +m4dd ;
                const T helper2 = -m5dd +m7dd ;

                C11d[i*cstr110+j*cstr111]  =  helper1 +helper2;
                C12d[i*cstr120+j*cstr121] = m3dd  + m5dd ;
                C21d[i*cstr210+j*cstr211] = m2dd  + m4dd ;

                T helper3 = m1dd - m2dd ;
                T helper4 = m3dd  + m6dd ;

                C22d[i*cstr220+j*cstr221]  =helper3+helper4;
            }
        }
    }
    else
    {
        #pragma omp parallel for simd collapse(2)
        for (size_t i = 0; i < half_n; i++)
        {
            for (size_t j = 0; j < half_p; j++)
            {
                const size_t mindex=i*str10+j*str11;
                const T m1dd=M1d[mindex];
                const T m2dd=M2d[mindex];
                const T m3dd=M3d[mindex];
                const T m4dd=M4d[mindex];
                const T m5dd=M5d[mindex];
                const T m6dd=M6d[mindex];
                const T m7dd=M7d[mindex];

                const T helper1 = m1dd  +m4dd ;
                const T helper2 = -m5dd +m7dd ;

                C11d[i*cstr110+j*cstr111]  =  helper1 +helper2;
                C12d[i*cstr120+j*cstr121] = m3dd  + m5dd ;
                C21d[i*cstr210+j*cstr211] = m2dd  + m4dd ;

                T helper3 = m1dd - m2dd ;
                T helper4 = m3dd  + m6dd ;

                C22d[i*cstr220+j*cstr221]  =helper3+helper4;
            }
        }
    }


    if(separate_device_memory)
    {
        omp_target_free(M1d,policy.devicenum);
        omp_target_free(M2d,policy.devicenum);
        omp_target_free(M3d,policy.devicenum);
        omp_target_free(M4d,policy.devicenum);
        omp_target_free(M5d,policy.devicenum);
        omp_target_free(M6d,policy.devicenum);
        omp_target_free(M7d,policy.devicenum);

        omp_target_free(Ard1,policy.devicenum);
        omp_target_free(Ard2,policy.devicenum);
        omp_target_free(Ard3,policy.devicenum);
        omp_target_free(Ard4,policy.devicenum);
        omp_target_free(Ard5,policy.devicenum);

        omp_target_free(Brd1,policy.devicenum);
        omp_target_free(Brd2,policy.devicenum);
        omp_target_free(Brd3,policy.devicenum);
        omp_target_free(Brd4,policy.devicenum);
        omp_target_free(Brd5,policy.devicenum);
    }

    else
    {
        if(policy.memmapped_files)
        {
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M1d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M2d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M3d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M4d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M5d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M6d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M7d,s);

            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(Ard1,s2);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(Ard2,s2);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(Ard3,s2);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(Ard4,s2);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(Ard5,s2);

            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(Brd1,s3);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(Brd2,s3);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(Brd3,s3);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(Brd4,s3);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(Brd5,s3);
        }
        else
        {
            delete[]M1d;
            delete[]M2d;
            delete[]M3d;
            delete[]M4d;
            delete[]M5d;
            delete[]M6d;
            delete[]M7d;
            delete[]Ard1;
            delete[]Ard2;
            delete[]Ard3;
            delete[]Ard4;
            delete[]Ard5;
            delete[]Brd1;
            delete[]Brd2;
            delete[]Brd3;
            delete[]Brd4;
            delete[]Brd5;
        }
    }

}

template <typename T>
void Math_Functions_MPI<T>::winograd_multiply(const DataBlock<T>& A, const DataBlock<T> &B, DataBlock<T>& C,const Math_MPI_RecursiveMultiplication_Policy*pol)
{
    const Math_MPI_RecursiveMultiplication_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    bool ongpu=policy.should_use_gpu(A,B,C,Math_Functions_Policy::default_cubic_treshold,7);
    bool separate_device_memory=false;
    if(ongpu)
    {


#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif
    }

    if(separate_device_memory)
    {
        typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadA(A, policy.devicenum, false);
        typename DataBlock_GPU_Memory_Functions<T>::OffloadHelperConst offloadB(B,  policy.devicenum, false);
        typename DataBlock_GPU_Memory_Functions<T>::OffloadHelper offloadC(C,  policy.devicenum, true, policy.update_host);
        DataBlock<T> tA=A,tB=B,tC=C;

        if(!tA.dpdata_is_devptr)
            tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);
        if(!tB.dpdata_is_devptr)
            tB.dpdata=(T*) omp_get_mapped_ptr(B.dpdata,policy.devicenum);
        if(!tC.dpdata_is_devptr)
            tC.dpdata=(T*) omp_get_mapped_ptr(C.dpdata,policy.devicenum);

        tA.dpdata_is_devptr=true;
        tB.dpdata_is_devptr=true;
        tC.dpdata_is_devptr=true;
        tA.devptr_devicenum=policy.devicenum;
        tB.devptr_devicenum=policy.devicenum;
        tC.devptr_devicenum=policy.devicenum;

        winograd_multiply_h(tA,tB,tC,ongpu, separate_device_memory,policy);
    }
    else
    {
        winograd_multiply_h(A,B,C,ongpu,false,policy);
    }

}

template <typename T>
void Math_Functions_MPI<T>::winograd_multiply_h(const DataBlock<T>& A,const DataBlock<T> &B, DataBlock<T>& C,bool ongpu, bool separate_device_memory, const Math_MPI_RecursiveMultiplication_Policy&policy)
{
    // Dimensions of input matrices
    size_t n = A.dpextents[0]; // Rows in A
    size_t m = A.dpextents[1]; // Columns in A and rows in B
    size_t p = A.dpextents[1]; // Columns in B


    if ((n%2!=0) || (m%2!=0) || (p%2!=0)  || m<=2 || n<=2|| p<=2 || !policy.should_use_recursion(n*p))
    {
        if(ongpu)
        {
            GPU_Math_Functions<T>::matrix_multiply_dot_g(   A,B,  C,policy.devicenum,false);
            return;
        }
        else
        {
            switch (policy.mode)
            {
            case Math_Functions_Policy::GPU_ONLY:
            {
                GPU_Math_Functions<T>::matrix_multiply_dot_g(   A,B,  C,policy.devicenum,true);
                return;
            }
            case Math_Functions_Policy::AUTO:
            {
                if(policy.should_use_gpu(A,B,C,Math_Functions_Policy::default_cubic_treshold,1))
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(A,B,C,policy.devicenum,true);
                else
                    In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w( A,B,C);
                return;

            }
            default:
            {
                In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w( A,B,  C);
                return;
            }
            }
        }
    }

    // Compute sizes for splitting

    size_t half_n = n / 2;
    size_t half_m = m / 2;
    size_t half_p = p / 2;

    // Submatrices of A

    size_t psext1[2],a11str[2],psext2[2],a12str[2],psext3[2],a21str[2],psext4[2],a22str[2],
           psext5[2],b11str[2],psext6[2],b12str[2],psext7[2],b21str[2],psext8[2],b22str[2];


    // Temporary storage for intermediate results
    size_t s=half_n*half_p;
    size_t s2=half_n*half_m;
    size_t s3=half_m*half_p;


    size_t ext1[2]= {half_n, half_p};
    size_t str1[2]= {half_p, 1};



    size_t ext2[2]= {half_n, half_m};
    size_t str2[2]= {half_m, 1};


    size_t ext3[2]=  {half_m, half_p};
    size_t str3[2]= {half_p, 1};



    T*S1d,*S2d,*S3d,*S4d,*S5d,*S6d,*S7d,*S8d,*M1d,*M2d,*M3d,*M4d,*M5d,*M6d,*M7d;
    if(separate_device_memory)
    {
        S1d=(T*)omp_target_alloc(sizeof(T)*s2, policy.devicenum);
        S2d=(T*)omp_target_alloc(sizeof(T)*s2, policy.devicenum);
        S3d=(T*)omp_target_alloc(sizeof(T)*s2, policy.devicenum);
        S4d=(T*)omp_target_alloc(sizeof(T)*s2, policy.devicenum);
        S5d=(T*)omp_target_alloc(sizeof(T)*s3, policy.devicenum);
        S6d=(T*)omp_target_alloc(sizeof(T)*s3, policy.devicenum);
        S7d=(T*)omp_target_alloc(sizeof(T)*s3, policy.devicenum);
        S8d=(T*)omp_target_alloc(sizeof(T)*s3, policy.devicenum);
        M1d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M2d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M3d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M4d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M5d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M6d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
        M7d=(T*)omp_target_alloc(sizeof(T)*s, policy.devicenum);
    }
    else
    {
        if(policy.memmapped_files)
        {
            S1d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s2);
            S2d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s2);
            S3d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s2);
            S4d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s2);
            S5d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s3);
            S6d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s3);
            S7d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s3);
            S8d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s3);
            M1d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M2d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M3d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M4d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M5d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M6d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
            M7d=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(s);
        }
        else
        {
            S1d=new T[s2];
            S2d=new T[s2];
            S3d=new T[s2];
            S4d=new T[s2];
            S5d=new T[s3];
            S6d=new T[s3];
            S7d=new T[s3];
            S8d=new T[s3];
            M1d=new T[s];
            M2d=new T[s];
            M3d=new T[s];
            M4d=new T[s];
            M5d=new T[s];
            M6d=new T[s];
            M7d=new T[s];
        }

    }


    DataBlock<T>
    S1(S1d,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
    S2(S2d,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
    S3(S3d,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),
    S4(S4d,s2,A.dprowmajor,2,ext2,str2,false,false,separate_device_memory),

    S5(S5d,s3,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
    S6(S6d,s3,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
    S7(S7d,s3,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),
    S8(S8d,s3,B.dprowmajor,2,ext3,str3,false,false,separate_device_memory),


    M1(M1d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M2(M2d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M3(M3d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M4(M4d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M5(M5d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M6(M6d,s,true,2,ext1,str1,false,false,separate_device_memory),
    M7(M7d,s,true,2,ext1,str1,false,false,separate_device_memory);



    DataBlock<T>  A11 = A.matrix_subspan(0, 0, half_n, half_m,psext1,a11str),
                  A12 = A.matrix_subspan(0, half_m, half_n, half_m,psext2,a12str),
                  A21 = A.matrix_subspan(half_n, 0, half_n, half_m,psext3,a21str),
                  A22 = A.matrix_subspan(half_n, half_m, half_n, half_m,psext4,a22str);

    // Submatrices of B
    DataBlock<T>  B11 = B.matrix_subspan(0, 0, half_m, half_p,psext5,b11str),
                  B12 = B.matrix_subspan(0, half_p, half_m, half_p,psext6,b12str),
                  B21 = B.matrix_subspan(half_m, 0, half_m, half_p,psext7,b21str),
                  B22 = B.matrix_subspan(half_m, half_p, half_m, half_p,psext8,b22str);


    const size_t a11str0=a11str[0];
    const size_t a11str1=a11str[1];

    const size_t a12str0=a12str[0];
    const size_t a12str1=a12str[1];

    const size_t a21str0=a21str[0];
    const size_t a21str1=a21str[1];

    const size_t a22str0=a22str[0];
    const size_t a22str1=a22str[1];

    const size_t b11str0=b11str[0];
    const size_t b11str1=b11str[1];

    const size_t b12str0=b12str[0];
    const size_t b12str1=b12str[1];

    const size_t b21str0=b21str[0];
    const size_t b21str1=b21str[1];

    const size_t b22str0=b22str[0];
    const size_t b22str1=b22str[1];

    const T* A11d=A11.dpdata;
    const T* A12d=A12.dpdata;
    const T* A21d=A21.dpdata;
    const T* A22d=A22.dpdata;

    const T* B11d=B11.dpdata;
    const T* B12d=B12.dpdata;
    const T* B21d=B21.dpdata;
    const T* B22d=B22.dpdata;

    const size_t strs0=str3[0];
    const size_t strs1=str3[1];

    if(ongpu)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum) is_device_ptr(A11d,A12d,A21d,A22d,S1d,S2d,S3d,S4d)
        for (size_t i=0; i<half_n; i++)
        {
            for (size_t j=0; j<half_m; j++)
            {

                const T a11dd=A11d[a11str0*i+a11str1*j];
                const T a12dd=A12d[a12str0*i+a12str1*j];
                const T a21dd=A21d[a21str0*i+a21str1*j];
                const T a22dd=A22d[a22str0*i+a22str1*j];

                const size_t sindex=strs0*i+strs1*j;

                const T s1=a21dd+a22dd;
                const T s2=s1-a11dd;

                S1d[sindex]=s1;

                S2d[sindex]=s2;
                S3d[sindex]=a11dd-a21dd;
                S4d[sindex]=a12dd-s2;

            }
        }
        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum)is_device_ptr(B11d,B12d,B21d,B22d,S5d,S6d,S7d,S8d)
        for (size_t i=0; i<half_m; i++)
        {
            for (size_t j=0; j<half_p; j++)
            {
                const T b11dd=B11d[b11str0*i+b11str1*j];
                const T b12dd=B12d[b12str0*i+b12str1*j];
                const T b21dd=B21d[b21str0*i+b21str1*j];
                const T b22dd=B22d[b22str0*i+b22str1*j];

                const size_t sindex=i*strs0+j*strs1;
                const T s5=b12dd-b11dd;
                const T s6=b22dd-s5;
                S5d[sindex]=s5;
                S6d[sindex]=b22dd-s5;
                S6d[sindex]=s6;
                S7d[sindex]=b22dd-b12dd;
                S8d[sindex]=s6-b21dd;
            }
        }
    }
    else
    {
        #pragma omp  parallel for simd collapse(2)
        for (size_t i=0; i<half_n; i++)
        {
            for (size_t j=0; j<half_m; j++)
            {
                const T a11dd=A11d[a11str0*i+a11str1*j];
                const T a12dd=A12d[a12str0*i+a12str1*j];
                const T a21dd=A21d[a21str0*i+a21str1*j];
                const T a22dd=A22d[a22str0*i+a22str1*j];

                const size_t sindex=strs0*i+strs1*j;

                const T s1=a21dd+a22dd;
                const T s2=s1-a11dd;

                S1d[sindex]=s1;

                S2d[sindex]=s2;
                S3d[sindex]=a11dd-a21dd;
                S4d[sindex]=a12dd-s2;

            }
        }
        #pragma omp parallel for simd collapse(2)
        for (size_t i=0; i<half_m; i++)
        {
            for (size_t j=0; j<half_p; j++)
            {
                const T b11dd=B11d[b11str0*i+b11str1*j];
                const T b12dd=B12d[b12str0*i+b12str1*j];
                const T b21dd=B21d[b21str0*i+b21str1*j];
                const T b22dd=B22d[b22str0*i+b22str1*j];

                const size_t sindex=i*strs0+j*strs1;
                const T s5=b12dd-b11dd;
                const T s6=b22dd-s5;
                S5d[sindex]=s5;
                S6d[sindex]=b22dd-s5;
                S6d[sindex]=s6;
                S7d[sindex]=b22dd-b12dd;
                S8d[sindex]=s6-b21dd;
            }
        }
    }



    if (policy.should_use_mpi_for_recursion(7))
    {
        int myrank=0,childdest=0;

        MPI_Comm_rank(policy.comm, &myrank);
        childdest=myrank*7;


        int message=Math_MPI_RecursiveMultiplication_Policy::WinogradVariant;

        MPI_Send(&message, 1, MPI_INT, childdest+1,0, policy.comm);
        size_t problemsize=s2;
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+1,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(S2,childdest+1,2,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(S6,childdest+1,3,policy.comm);


        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+2,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+2,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(A11,childdest+2,2,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(B11,childdest+2,3,policy.comm);



        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+3,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+3,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(A12,childdest+3,2,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(B21,childdest+3,3,policy.comm);

        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+4,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+4,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(S3,childdest+4,2,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(S7,childdest+4,3,policy.comm);

        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+5,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+5,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(S1,childdest+5,2,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(S5,childdest+5,3,policy.comm);

        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+6,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+6,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(S4,childdest+6,2,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(B22,childdest+6,3,policy.comm);

        problemsize=s2;
        MPI_Send(&message, 1, MPI_INT, childdest+7,0, policy.comm);
        MPI_Send(&problemsize, 1, mpi_get_type<size_t>(), childdest+7,1, policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(A22,childdest+7,2,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Send_DataBlock(S8,childdest+7,3,policy.comm);


        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M1,childdest+1,4,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M2,childdest+2,4,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M3,childdest+3,4,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M4,childdest+4,4,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M5,childdest+5,4,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M6,childdest+6,4,policy.comm);
        DataBlock_MPI_Functions<T>::MPI_Recv_DataBlock_pdata(M7,childdest+7,4,policy.comm);

    }
    else
    {

        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                {
                    winograd_multiply_h(S2,S6,M1, ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(A11,B11,M2, ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(A12,B21,M3, ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(S3,S7,M4,ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(S1,S5,M5,ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(S4,B22,M6,ongpu,  separate_device_memory,policy);
                }
                #pragma omp task
                {
                    winograd_multiply_h(A22,S8,M7,ongpu,  separate_device_memory,policy);
                }
                #pragma omp taskwait
            }

        }

    }


    size_t pext10a[2],cstr11[2],pext11a[2],cstr12[2],pext12a[2],cstr21[2],pext13a[2],cstr22[2];

    DataBlock<T>  C11 = C.matrix_subspan(0, 0, half_n, half_p,pext10a,cstr11),
                  C12 = C.matrix_subspan(0, half_p, half_n, half_p,pext11a,cstr12),
                  C21 = C.matrix_subspan(half_n, 0, half_n, half_p,pext12a,cstr21),
                  C22 = C.matrix_subspan(half_n, half_p, half_n, half_p,pext13a,cstr22);

    const size_t cstr110=cstr11[0];
    const size_t cstr111=cstr11[1];

    const size_t cstr120=cstr12[0];
    const size_t cstr121=cstr12[1];

    const size_t cstr210=cstr21[0];
    const size_t cstr211=cstr21[1];

    const size_t cstr220=cstr22[0];
    const size_t cstr221=cstr22[1];
    T* C11d=C11.dpdata;
    T* C12d=C12.dpdata;
    T* C21d=C21.dpdata;
    T* C22d=C22.dpdata;


    const size_t str10=str1[0];
    const size_t str11=str1[1];

    if(ongpu)
    {
        #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum) is_device_ptr(M1d,M2d,M3d,M4d,M5d,M6d,M7d,C11d,C12d,C21d,C22d)
        for (size_t i = 0; i < half_n; ++i)
        {
            for (size_t j = 0; j < half_p; ++j)
            {
                const size_t mindex=i*str10+j*str11;
                const T m1dd=M1d[mindex];
                const T m2dd=M2d[mindex];
                const T m3dd=M3d[mindex];
                const T m4dd=M4d[mindex];
                const T m5dd=M5d[mindex];
                const T m6dd=M6d[mindex];
                const T m7dd=M7d[mindex];

                const T T1=m1dd+m2dd;
                const T T2=T1+m4dd;
                const T helper=m5dd+m6dd;
                C11d[cstr110*i+cstr111*j] = m2dd + m3dd;
                C12d[cstr120*i+cstr121*j] = T1 +helper ;
                C21d[cstr210*i+cstr211*j] = T2 - m7dd;
                C22d[cstr220*i+cstr221*j] = T2 + m5dd;
            }
        }
    }
    else
    {
        #pragma omp parallel for simd collapse(2)
        for (size_t i = 0; i < half_n; ++i)
        {
            for (size_t j = 0; j < half_p; ++j)
            {
                const size_t mindex=i*str10+j*str11;
                const T m1dd=M1d[mindex];
                const T m2dd=M2d[mindex];
                const T m3dd=M3d[mindex];
                const T m4dd=M4d[mindex];
                const T m5dd=M5d[mindex];
                const T m6dd=M6d[mindex];
                const T m7dd=M7d[mindex];

                const T T1=m1dd+m2dd;
                const T T2=T1+m4dd;
                const T helper=m5dd+m6dd;
                C11d[cstr110*i+cstr111*j] = m2dd + m3dd;
                C12d[cstr120*i+cstr121*j] = T1 +helper ;
                C21d[cstr210*i+cstr211*j] = T2 - m7dd;
                C22d[cstr220*i+cstr221*j] = T2 + m5dd;
            }
        }

    }


    if(separate_device_memory)
    {
        omp_target_free(M1d,policy.devicenum);
        omp_target_free(M2d,policy.devicenum);
        omp_target_free(M3d,policy.devicenum);
        omp_target_free(M4d,policy.devicenum);
        omp_target_free(M5d,policy.devicenum);
        omp_target_free(M6d,policy.devicenum);
        omp_target_free(M7d,policy.devicenum);

        omp_target_free(S1d,policy.devicenum);
        omp_target_free(S2d,policy.devicenum);
        omp_target_free(S3d,policy.devicenum);
        omp_target_free(S4d,policy.devicenum);
        omp_target_free(S5d,policy.devicenum);

        omp_target_free(S6d,policy.devicenum);
        omp_target_free(S7d,policy.devicenum);
        omp_target_free(S8d,policy.devicenum);

    }
    else
    {

        if(policy.memmapped_files)
        {
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M1d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M2d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M3d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M4d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M5d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M6d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(M7d,s);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(S1d,s2);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(S2d,s2);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(S3d,s2);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(S4d,s2);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(S5d,s3);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(S6d,s3);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(S7d,s3);
            DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(S8d,s3);
        }
        else
        {
            delete[]M1d;
            delete[]M2d;
            delete[]M3d;
            delete[]M4d;
            delete[]M5d;
            delete[]M6d;
            delete[]M7d;
            delete[]S1d;
            delete[]S2d;
            delete[]S3d;
            delete[]S4d;
            delete[]S5d;
            delete[]S6d;
            delete[]S7d;
            delete[]S8d;
        }
    }
}

template <typename T>
void Math_Functions_MPI<T>::cholesky_decomposition(const DataBlock<T> & A,DataBlock<T> & L, Math_MPI_Decomposition_Policy *pol)
{
    Math_MPI_Decomposition_Policy policy = (pol != nullptr) ? *pol : get_default_policy();
    Math_Functions_MPI<T>::cholesky_decomposition_h(A,L,policy);
}

template <typename T>
void Math_Functions_MPI<T>::cholesky_decomposition_h(const DataBlock<T> & A,DataBlock<T> & L, Math_MPI_Decomposition_Policy &policy)
{


    bool ongpu=policy.should_use_gpu(A,L,Math_Functions_Policy::default_cubic_treshold,1);
    bool separate_device_memory=false;
    if(ongpu)
    {
#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif
    }



    const size_t n = A.dpextents[0];

    size_t step_size=policy.step_size;

    if(step_size==0)
        step_size=(size_t)pow(n,0.8385);

    if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    size_t tempsize=(n-step_size)*(n-step_size);


    if(ongpu)
    {
        T * sdata;
        T* tempad;

        if(separate_device_memory)
        {
            sdata= (T*) omp_target_alloc(sizeof(T)*tempsize, policy.devicenum);
            tempad= (T*) omp_target_alloc(sizeof(T)*A.dpdatalength, policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                sdata=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(tempsize);
                tempad=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(A.dpdatalength);
            }
            else
            {
                sdata=new T[tempsize];
                tempad=new T[A.dpdatalength];
            }
        }
        size_t aext[2]= {A.dpextents[0],A.dpextents[1]};
        size_t astr[2]= {A.dpstrides[0],A.dpstrides[1]};

        DataBlock<T> tempA(tempad,A.dpdatalength,A.dprowmajor,2,aext,astr,false,false,separate_device_memory,policy.devicenum);

        DataBlock<T> tA=A,tL=L;


        if(separate_device_memory)
        {
            DataBlock_GPU_Memory_Functions<T>::create_in(A,policy.devicenum);
            DataBlock_GPU_Memory_Functions<T>::create_out(L,policy.devicenum);

            if(!A.dpdata_is_devptr)
                tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);


            if(!L.dpdata_is_devptr)
                tL.dpdata=(T*) omp_get_mapped_ptr(L.dpdata,policy.devicenum);


            tA.dpdata_is_devptr=true;
            tL.dpdata_is_devptr=true;
            tA.devptr_devicenum=policy.devicenum;
            tL.devptr_devicenum=policy.devicenum;
        }

        const size_t Lstr0=tL.dpstrides[0];
        const size_t Lstr1=tL.dpstrides[1];
        const size_t Astr0=tA.dpstrides[0];
        const size_t Astr1=tA.dpstrides[1];

        if(policy.initialize_output_to_zeros)
        {

            #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    tL.dpdata[i*Lstr0+j*Lstr1]=0;
                    tempA.dpdata[i*Astr0+j*Astr1]=tA.dpdata[i*Astr0+j*Astr1];
                }
            }
        }
        else
        {
            #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    tempA.dpdata[i*Astr0+j*Astr1]=tA.dpdata[i*Astr0+j*Astr1];
                }
            }
        }

        size_t z=0;
        for (size_t c = 0; c < n; ++c)   // Iterate over columns
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;
                size_t sub_ext[2];
                size_t sub_str[2];
                DataBlock<T> R = tL.matrix_subspan(c, z,u,v,sub_ext,sub_str);

                size_t sextt[2]= {u,u};
                size_t sstrt[2]= {u,1};
                DataBlock<T>  S(sdata,u*u,true,2,sextt,sstrt,false,false,true,tA.devptr_devicenum);

                size_t rtext[2],strtext[2];

                DataBlock<T> RT=R.matrix_transpose(rtext,strtext);

                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(R,RT,S,policy.devicenum,false);
                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(R,RT,S,ongpu,separate_device_memory,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(R,RT,S,ongpu,separate_device_memory,policy);
                    break;
                }

                #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum)
                for (size_t i = c; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA.dpdata[i*Astr0+j*Astr1] -= S.dpdata[(i - c)*sstrt[0]+ (j - c)*sstrt[1]];
                    }
                }

                z = c;
            }


            T tmp=T(0);

            #pragma omp target teams distribute parallel for simd map(tofrom:tmp) reduction(+:tmp)  device(policy.devicenum)
            for (size_t k = z; k < c; ++k)
            {
                const T tmp3=tL.dpdata[c*Lstr0+k*Lstr1];
                tmp+= tmp3 * tmp3;
            }
            T tmp42=T(0);
            omp_target_memcpy(&tmp42,&tempA.dpdata[0],sizeof(T),0,sizeof(T)*(Astr0*c+Astr1*c),omp_get_initial_device(),policy.devicenum);
            tmp=tmp42-tmp;

            const T temp4=sqrt(tmp);

            omp_target_memcpy(&tL.dpdata[0],&temp4,sizeof(T),sizeof(T)*(Lstr0*c+Lstr1*c),0,policy.devicenum,omp_get_initial_device());

            #pragma omp target teams distribute parallel for map(to:temp4) device(policy.devicenum)
            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 = T(0);
                #pragma omp simd reduction(+:tmp2)
                for (size_t k = z; k < c; ++k)
                {
                    tmp2 += tL.dpdata[i*Lstr0+k*Lstr1] * tL.dpdata[c*Lstr0+k*Lstr1];
                }
                tmp2=tempA.dpdata[i*Astr0+c*Astr1]-tmp2;
                tL.dpdata[i*Lstr0+ c*Lstr1]=tmp2/temp4;
            }
        }

        if(separate_device_memory)
        {
            if(policy.update_host)
                DataBlock_GPU_Memory_Functions<T>::update_host(L,policy.devicenum);
            DataBlock_GPU_Memory_Functions<T>::release(L,policy.devicenum);
            DataBlock_GPU_Memory_Functions<T>::release(A,policy.devicenum);

            omp_target_free(sdata,  policy.devicenum);
            omp_target_free(tempad, policy.devicenum);

        }
        else
        {
            if(policy.memmapped_files)
            {
                DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(sdata,tempsize);
                DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(tempad,A.dpdatalength);
            }
            else
            {
                delete[] sdata;
                delete[] tempad;
            }
        }

    }
    else
    {

        T * sdata= DataBlock_Host_Memory_Functions<T>::alloc_data_ptr(tempsize,policy.memmapped_files);

        DataBlock<T>  tempA=DataBlock_Host_Memory_Functions<T>::alloc_data_copy_strides_extents(A.dpdatalength,A.dprowmajor, A.dprank,A.dpextents,A.dpstrides,policy.memmapped_files);

        if (policy.initialize_output_to_zeros)
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    L(i,j)=0;
                    tempA(i,j)=A(i,j);
                }
            }
        }
        else
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    tempA(i,j)=A(i,j);
                }
            }
        }


        size_t z=0;
        for (size_t c = 0; c < n; ++c)   // Iterate over columns
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;
                size_t sub_ext[2];
                size_t sub_str[2];
                DataBlock<T> R = L.matrix_subspan(c, z,u,v,sub_ext,sub_str);

                size_t sextt[2]= {u,u};
                size_t sstrt[2]= {u,1};
                DataBlock<T>  S(sdata,u*u,true,2,sextt,sstrt,false,false,false);

                size_t rtext[2],strtext[2];

                DataBlock<T> RT=R.matrix_transpose(rtext,strtext);


                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    if(policy.should_use_gpu(R,RT,S,Math_Functions_Policy::default_cubic_treshold,1))
                        GPU_Math_Functions<T>::matrix_multiply_dot_g(R,RT,S,policy.devicenum,true);
                    else
                        In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(R,RT,S);
                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(R,RT,S,ongpu,separate_device_memory,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(R,RT,S,ongpu,separate_device_memory,policy);
                }


                #pragma omp parallel for simd collapse(2)
                for (size_t i = c; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i, j) -= S(i - c, j - c);
                    }
                }


                z = c;
            }


            T tmp=T(0);

            #pragma omp parallel for simd  reduction(+: tmp)
            for (size_t k = z; k < c; ++k)
            {
                const T tmp3=L(c,k);
                tmp+= tmp3 * tmp3;
            }
            tmp=tempA(c, c)-tmp;
            T tmp4=sqrt(tmp);
            L(c, c)=tmp4;

            #pragma omp parallel for
            for (size_t i = c + 1; i < n; ++i)
            {
                T tmp2 = T(0);
                #pragma omp simd reduction(+:tmp2)
                for (size_t k = z; k < c; ++k)
                {
                    tmp2 += L(i, k) * L(c, k);
                }

                tmp2=tempA(i, c)-tmp2;

                L(i, c)=tmp2/tmp4;
            }
        }
        DataBlock_Host_Memory_Functions<T>::free_copy(tempA,policy.memmapped_files);
        DataBlock_Host_Memory_Functions<T>::free_data_ptr(sdata,tempsize,policy.memmapped_files);
    }
}


template <typename T>
void Math_Functions_MPI<T>::lu_decomposition(const DataBlock<T>& A, DataBlock<T> &L,DataBlock<T>& U,  Math_MPI_Decomposition_Policy* pol)
{
    Math_MPI_Decomposition_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    Math_Functions_MPI<T>::lu_decomposition_h(A,L,U,policy);


}
template <typename T>
void Math_Functions_MPI<T>::lu_decomposition_h(const DataBlock<T>& A, DataBlock<T> &L,DataBlock<T>& U,  Math_MPI_Decomposition_Policy& policy)
{

    bool ongpu=policy.should_use_gpu(A,L,U,Math_Functions_Policy::default_cubic_treshold,1);

    size_t n = A.dpextents[0];
    int step_size=policy.step_size;

    if(step_size==0)
        step_size=(size_t)pow(n,0.8385);
    if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    size_t tempsize=(n-step_size)*(n-step_size);

    if(ongpu)
    {
        bool separate_device_memory=false;
#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif

        T * sdata;
        T* tempad;

        if(separate_device_memory)
        {
            sdata= (T*) omp_target_alloc(sizeof(T)*tempsize, policy.devicenum);
            tempad= (T*) omp_target_alloc(sizeof(T)*A.dpdatalength, policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                sdata=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(tempsize);
                tempad=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(A.dpdatalength);
            }
            else
            {
                sdata=new T[tempsize];
                tempad=new T[A.dpdatalength];
            }
        }

        size_t taext[2]= {A.dpextents[0],A.dpextents[1]};
        size_t tastr[2]= {A.dpstrides[0],A.dpstrides[1]};

        DataBlock<T> tempA(tempad,A.dpdatalength,A.dprowmajor,2,taext,tastr,false,false,separate_device_memory,policy.devicenum);

        DataBlock<T> tA=A,tL=L,tU=U;


        if(separate_device_memory)
        {
            DataBlock_GPU_Memory_Functions<T>::create_in(A,policy.devicenum);
            DataBlock_GPU_Memory_Functions<T>::create_out(L,policy.devicenum);
            DataBlock_GPU_Memory_Functions<T>::create_out(U,policy.devicenum);

            if(!A.dpdata_is_devptr)
                tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);
            if(!L.dpdata_is_devptr)
                tL.dpdata=(T*) omp_get_mapped_ptr(L.dpdata,policy.devicenum);
            if(!U.dpdata_is_devptr)
                tU.dpdata=(T*) omp_get_mapped_ptr(U.dpdata,policy.devicenum);

            tA.dpdata_is_devptr=true;
            tL.dpdata_is_devptr=true;
            tU.dpdata_is_devptr=true;
            tA.devptr_devicenum=policy.devicenum;
            tL.devptr_devicenum=policy.devicenum;
            tU.devptr_devicenum=policy.devicenum;

        }

        const size_t Astr0=tA.dpstrides[0];
        const size_t Astr1=tA.dpstrides[1];
        const size_t Lstr0=tL.dpstrides[0];
        const size_t Lstr1=tL.dpstrides[1];
        const size_t Ustr0=tU.dpstrides[0];
        const size_t Ustr1=tU.dpstrides[1];

        if(policy.initialize_output_to_zeros)
        {
            #pragma omp target teams distribute parallel for simd collapse(2)  device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    tL.dpdata[i*Lstr0+j*Lstr1]=0;
                    tU.dpdata[i*Ustr0+j*Ustr1]=0;
                    tempA.dpdata[i*Astr0+j*Astr1]=tA.dpdata[i*Astr0+j*Astr1];
                }
            }
        }
        else
        {
            #pragma omp target teams distribute parallel for simd collapse(2)   device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    tempA.dpdata[i*Astr0+j*Astr1]=tA.dpdata[i*Astr0+j*Astr1];
                }
            }
        }

        size_t z=0;

        for (size_t c = 0; c < n; ++c)
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;

                size_t sub_ext[2];
                size_t sub_str[2];
                DataBlock<T> RL = tL.matrix_subspan(c, z,u, v,sub_ext,sub_str);
                size_t sub_ext2[2];
                size_t sub_str2[2];
                DataBlock<T> RU = tU.matrix_subspan(z, c,v, u,sub_ext2,sub_str2);

                size_t sextt[2]= {u,u};
                size_t sstrt[2]= {u,1};

                DataBlock<T>  S(sdata,u*u,true,2,sextt,sstrt,false,false,separate_device_memory,tA.devptr_devicenum);

                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                {
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(RL,RU,S,policy.devicenum,false);
                    break;
                }
                case Math_MPI_Decomposition_Policy::Strassen:
                {
                    strassen_multiply_h(RL,RU,S,ongpu,separate_device_memory, policy);
                    break;
                }
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                {
                    winograd_multiply_h(RL,RU,S,ongpu, separate_device_memory,policy);
                    break;
                }
                }

                #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum)
                for (size_t i = c; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA.dpdata[i*Astr0+j*Astr1] -= S.dpdata[(i - c)*sstrt[0]+(j - c)*sstrt[1]];
                    }
                }

                z = c;
            }


            #pragma omp target teams distribute device(policy.devicenum)
            for (size_t i = c; i < n; ++i)
            {
                T temp=T(0);
                #pragma omp parallel for simd reduction(+:temp)
                for (size_t k = z; k < c; ++k)
                {
                    temp += tU.dpdata[ k*Ustr0+i*Ustr1] * tL.dpdata[ c*Lstr0+k*Lstr1];
                }
                temp=tempA.dpdata[c*Astr0+i*Astr1]-temp;
                tU.dpdata[c*Ustr0+i*Ustr1]=temp;
            }

            T temp4=T(0);
            omp_target_memcpy(&temp4,&tU.dpdata[0],sizeof(T),0,sizeof(T)*(Ustr0*c+Ustr1*c),omp_get_initial_device(),policy.devicenum);


            #pragma omp target teams distribute device(policy.devicenum)
            for (size_t i = c; i < n; ++i)
            {
                T temp = T(0);
                #pragma omp parallel for simd reduction(+:temp)
                for (size_t k = z; k < c; ++k)
                {
                    temp += tU.dpdata[k*Ustr0+c*Ustr1] * tL.dpdata[i*Lstr0+k*Lstr1];
                }
                temp=tempA.dpdata[i*Astr0+c*Astr1]-temp;
                tL.dpdata[i*Lstr0+c*Lstr1]=temp/temp4;
            }
        }


        if(separate_device_memory)
        {
            DataBlock_GPU_Memory_Functions<T>::release(A,policy.devicenum);
            if(policy.update_host)
            {
                DataBlock_GPU_Memory_Functions<T>::update_host(L,policy.devicenum);
                DataBlock_GPU_Memory_Functions<T>::update_host(U,policy.devicenum);
            }

            omp_target_free(sdata,  policy.devicenum);
            omp_target_free(tempad, policy.devicenum);

            DataBlock_GPU_Memory_Functions<T>::release(L,policy.devicenum);
            DataBlock_GPU_Memory_Functions<T>::release(U,policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(sdata,tempsize);
                DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(tempad,A.dpdatalength);
            }
            else
            {
                delete[] sdata;
                delete[] tempad;
            }
        }

    }
    else
    {

        T * sdata= DataBlock_Host_Memory_Functions<T>::alloc_data_ptr(tempsize,policy.memmapped_files);

        DataBlock<T>  tempA=DataBlock_Host_Memory_Functions<T>::alloc_data_copy_strides_extents(A.dpdatalength,A.dprowmajor, A.dprank,A.dpextents,A.dpstrides
                            ,policy.memmapped_files);

        if (policy.initialize_output_to_zeros)
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    L(i,j)=0;
                    U(i,j)=0;
                    tempA(i,j)=A(i,j);
                }
            }
        }
        else
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j <n; ++j)
                {
                    tempA(i,j)=A(i,j);
                }
            }

        }

        size_t z=0;
        for (size_t c = 0; c < n; ++c)
        {
            if (c == z + step_size)
            {
                size_t u=n-c;
                size_t v=c-z;

                size_t sub_ext[2];
                size_t sub_str[2];
                DataBlock<T> RL = L.matrix_subspan(c, z,u, v,sub_ext,sub_str);
                size_t sub_ext2[2];
                size_t sub_str2[2];
                DataBlock<T> RU = U.matrix_subspan(z, c,v, u,sub_ext2,sub_str2);

                size_t sextt[2]= {u,u};
                size_t sstrt[2]= {u,1};
                DataBlock<T>  S(sdata,u*u,true,2,sextt,sstrt,false,false,false);




                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                {
                    if(policy.should_use_gpu(RL,RU,S,Math_Functions_Policy::default_cubic_treshold,1))
                    {
                        GPU_Math_Functions<T>::matrix_multiply_dot_g(RL,RU,S,policy.devicenum,true);
                    }
                    else
                    {
                        In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(RL,RU,S);
                    }
                    break;
                }
                case Math_MPI_Decomposition_Policy::Strassen:
                {
                    strassen_multiply_h(RL,RU,S,false,false,policy);
                    break;
                }
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                {
                    winograd_multiply_h(RL,RU,S,false,false,policy);
                    break;
                }
                }

                #pragma omp parallel for simd collapse(2)
                for (size_t i = c; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        tempA(i,j) -= S(i - c, j - c);
                    }
                }
                z = c;
            }

            #pragma omp parallel for
            for (size_t i = c; i < n; ++i)
            {
                T temp=T(0);
                #pragma omp simd reduction(+:temp)
                for (size_t k = z; k < c; ++k)
                {
                    temp += U( k,i) * L( c,k);
                }
                temp=tempA(c,i)-temp;
                U(c,i)=temp;
            }

            const T temp4=U(c,c);

            #pragma omp parallel for
            for (size_t i = c; i < n; ++i)
            {
                T temp = 0;
                #pragma omp simd reduction(+:temp)
                for (size_t k = z; k < c; ++k)
                {
                    temp += U(k,c) * L( i,k);
                }
                temp=tempA(i,c)-temp;
                L(i,c)=temp/temp4;
            }
        }

        DataBlock_Host_Memory_Functions<T>::free_copy(tempA,policy.memmapped_files);
        DataBlock_Host_Memory_Functions<T>::free_data_ptr(sdata,tempsize,policy.memmapped_files);
    }

}

template <typename T>
void Math_Functions_MPI<T>::qr_decomposition(const DataBlock<T>& A, DataBlock<T>& Q, DataBlock<T>& R, Math_MPI_Decomposition_Policy *pol)
{

    Math_MPI_Decomposition_Policy policy = (pol != nullptr) ? *pol : get_default_policy();
    Math_Functions_MPI<T>::qr_decomposition_h(A,Q,R,policy);

}
template <typename T>
void Math_Functions_MPI<T>::qr_decomposition_h(const DataBlock<T>& A, DataBlock<T>& Q, DataBlock<T>& R, Math_MPI_Decomposition_Policy &policy)
{

    bool ongpu=policy.should_use_gpu(A,Q,R,Math_Functions_Policy::default_cubic_treshold,1);

    int step_size=policy.step_size;

    if(step_size==0)
        step_size=(size_t)pow(A.dpextents[0],0.8385);

    if (step_size% 2 !=0 &&step_size>=1)
        step_size=step_size-1;

    size_t n = A.dpextents[0]; // Number of rows (assuming 2D matrix)
    size_t m = A.dpextents[1]; // Number of columns

    // Initialize Q and R matrices
    size_t nm=n*m, mm=m*m;
    if(ongpu)
    {


        bool separate_device_memory=false;
#if !defined(Unified_Shared_Memory)
        separate_device_memory=true;
#endif

        T * tempC;
        T * tempS;
        T*  tempM;
        if(separate_device_memory)
        {
            tempS= (T*) omp_target_alloc(sizeof(T)*nm, policy.devicenum);
            tempC= (T*) omp_target_alloc(sizeof(T)*mm, policy.devicenum);
            tempM= (T*) omp_target_alloc(sizeof(T)*A.dpdatalength, policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                tempS=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(nm);
                tempC=DataBlock_Host_Memory_Functions<T>::create_temp_mmap(mm);
                tempM= DataBlock_Host_Memory_Functions<T>::create_temp_mmap(A.dpdatalength);
            }
            else
            {
                tempS=new T[nm];
                tempC=new T[mm];
                tempM=new T[A.dpdatalength];
            }
        }
        size_t aext[2]= {A.dpextents[0],A.dpextents[1]};
        size_t astr[2]= {A.dpstrides[0],A.dpstrides[1]};
        DataBlock<T> M(tempM,A.dpdatalength,A.dprowmajor,2,aext,astr,false,false,separate_device_memory,policy.devicenum);

        DataBlock<T> tA=A,tQ=Q,tR=R;


        if(separate_device_memory)
        {
            DataBlock_GPU_Memory_Functions<T>::create_in(A,policy.devicenum);
            DataBlock_GPU_Memory_Functions<T>::create_out(Q,policy.devicenum);
            DataBlock_GPU_Memory_Functions<T>::create_out(R,policy.devicenum);


            if(!A.dpdata_is_devptr)
                tA.dpdata=(T*) omp_get_mapped_ptr(A.dpdata,policy.devicenum);
            if(!Q.dpdata_is_devptr)
                tQ.dpdata=(T*) omp_get_mapped_ptr(Q.dpdata,policy.devicenum);
            if(!R.dpdata_is_devptr)
                tR.dpdata=(T*) omp_get_mapped_ptr(R.dpdata,policy.devicenum);

            tA.dpdata_is_devptr=true;
            tQ.dpdata_is_devptr=true;
            tR.dpdata_is_devptr=true;
            tA.devptr_devicenum=policy.devicenum;
            tQ.devptr_devicenum=policy.devicenum;
            tR.devptr_devicenum=policy.devicenum;
        }

        const size_t Qstr0=Q.dpstrides[0];
        const size_t Qstr1=Q.dpstrides[1];
        const size_t Rstr0=R.dpstrides[0];
        const size_t Rstr1=R.dpstrides[1];
        const size_t Astr0=A.dpstrides[0];
        const size_t Astr1=A.dpstrides[1];

        if(policy.initialize_output_to_zeros)
        {

            #pragma omp target teams distribute parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    tQ.dpdata[i*Qstr0 + j*Qstr1] = T(0);
                }
            }

            #pragma omp target teams distribute parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < m; ++j)
                {
                    M.dpdata[i*Astr0 + j*Astr1] = tA.dpdata[i*Astr0 + j*Astr1];
                    tR.dpdata[i*Rstr0 + j*Rstr1] = T(0);
                }
            }
        }
        else
        {
            #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < m; ++j)
                {
                    M.dpdata[i*Astr0+j*Astr1]=tA.dpdata[i*Astr0+j*Astr1];
                }
            }
        }




        size_t z = 0;

        for (size_t c = 0; c < m; ++c)
        {

            if (c == z +step_size)
            {

                size_t cz=c-z;
                size_t mc=m-c;
                // Extract submatrices

                size_t extBQ[2],strBQ[2];

                size_t extBM[2],strBM[2];

                DataBlock<T> BQ = tQ.matrix_subspan(0, z, n, cz,extBQ,strBQ);
                DataBlock<T> BM = M.matrix_subspan(0, c, n,mc,extBM,strBM);

                size_t tempCextt[2]= {cz,mc};
                size_t tempCstrt[2]= {mc,1};

                DataBlock<T>  C(tempC,cz*mc,true,2,tempCextt,tempCstrt,false,false,separate_device_memory,tA.devptr_devicenum);


                size_t extBQT[2],strBQT[2];
                DataBlock<T> BQT=BQ.matrix_transpose(extBQT,strBQT);


                GPU_Math_Functions<T>::matrix_multiply_dot_g(BQT,BM,C,policy.devicenum,false);

                size_t sextt[2]= {n,mc};
                size_t sstrt[2]= {mc,1};
                DataBlock<T>  S(tempS,n*mc,true,2,sextt,sstrt,false,false,separate_device_memory,tA.devptr_devicenum);

                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(BQ,C,S,policy.devicenum,false);
                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(BQ,C,S,ongpu,separate_device_memory,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(BQ,C,S,ongpu,separate_device_memory,policy);
                    break;
                }


                #pragma omp target teams distribute parallel for simd collapse(2) device(policy.devicenum)
                for (size_t i = 0; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        M.dpdata[i*Astr0+j*Astr1] -= S.dpdata[i*sstrt[0]+(j-c)*sstrt[1]];
                    }
                }
                z = c;
            }
//            // Extract column c of M

            size_t vext[1],vstr[1];
            DataBlock<T> v = M.matrix_column(c,vext,vstr);
            const size_t pextv0=vext[0];
            for (size_t j = z; j < c; ++j)
            {
                size_t uext[1],ustr[1];
                DataBlock<T>  u = tQ.matrix_column(j,uext,ustr);

                T dot_pr=T(0);

                #pragma omp target teams distribute parallel for simd  map(tofrom: dot_pr) reduction(+:dot_pr) device(policy.devicenum)
                for (size_t i = 0; i < pextv0; ++i)
                {
                    dot_pr += u.dpdata[i*ustr[0]] * v.dpdata[i*vstr[0]];
                }

                const T cdot_pr = dot_pr;
                #pragma omp target teams distribute parallel for simd device(policy.devicenum)
                for (size_t i = 0; i < pextv0; ++i)
                {
                    v.dpdata[i*vstr[0]] -= cdot_pr * u.dpdata[i*ustr[0]];
                }

            }

            T norm = T(0);
            #pragma omp target  teams distribute parallel for simd map(tofrom:norm) reduction(+:norm)device(policy.devicenum)
            for (size_t i = 0; i < pextv0; ++i)
            {
                norm += v.dpdata[i*vstr[0]] * v.dpdata[i*vstr[0]];
            }

            const T normc = sqrt(norm);

            #pragma omp target teams distribute parallel for simd device(policy.devicenum)
            for (size_t i = 0; i < pextv0; ++i)
            {
                tQ.dpdata[i*Qstr0+c*Qstr1] = v.dpdata[i*vstr[0]]/normc;
            }

        }
        // Compute R = Q^T * A
        size_t extQT[2],strQT[2];

        DataBlock<T> QT=tQ.matrix_transpose(extQT,strQT);


        switch (policy.algorithm_version)
        {
        case Math_MPI_Decomposition_Policy::Naive:
            GPU_Math_Functions<T>::matrix_multiply_dot_g(QT,tA,tR,policy.devicenum,false);
            break;
        case Math_MPI_Decomposition_Policy::Strassen:
            strassen_multiply_h(QT,tA,tR,ongpu,separate_device_memory,policy);
            break;
        case Math_MPI_Decomposition_Policy::WinogradVariant:
            winograd_multiply_h(QT,tA,tR,ongpu,separate_device_memory,policy);
            break;
        }

        if(separate_device_memory)
        {
            if(policy.update_host)
            {
                DataBlock_GPU_Memory_Functions<T>::update_host(Q,policy.devicenum);
                DataBlock_GPU_Memory_Functions<T>::update_host(R,policy.devicenum);
            }
            DataBlock_GPU_Memory_Functions<T>::release(A,policy.devicenum);
            DataBlock_GPU_Memory_Functions<T>::release(Q,policy.devicenum);
            DataBlock_GPU_Memory_Functions<T>::release(R,policy.devicenum);

            omp_target_free(tempS, policy.devicenum);
            omp_target_free(tempC, policy.devicenum);
            omp_target_free(tempM, policy.devicenum);
        }
        else
        {
            if(policy.memmapped_files)
            {
                DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(tempS,nm);
                DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(tempM,A.dpdatalength);
                DataBlock_Host_Memory_Functions<T>::delete_temp_mmap(tempC,mm);
            }
            else
            {
                delete[] tempS;
                delete[] tempM;
                delete[] tempC;
            }
        }


    }
    else
    {


        DataBlock<T> M= DataBlock_Host_Memory_Functions<T>::alloc_data_copy_strides_extents(A.dpdatalength,A.dprowmajor,
                        A.dprank,A.dpextents,A.dpstrides,
                        policy.memmapped_files);

        T * tempC= DataBlock_Host_Memory_Functions<T>::alloc_data_ptr(mm,policy.memmapped_files);
        T * tempS= DataBlock_Host_Memory_Functions<T>::alloc_data_ptr(nm,policy.memmapped_files);


        if(policy.initialize_output_to_zeros)
        {
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i)
            {
                #pragma omp simd
                for (size_t j = 0; j < n; ++j)
                    Q(i,j) = 0;

                #pragma omp simd
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                    R(i,j) = T(0);
                }
            }
        }
        else
        {
            #pragma omp parallel for simd collapse(2)
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < m; ++j)
                {
                    M(i,j)=A(i,j);
                }
            }
        }


        size_t z = 0;

        for (size_t c = 0; c < m; ++c)
        {
            if (c == z +step_size)
            {
                size_t cz=c-z;
                size_t mc=m-c;
                // Extract submatrices

                size_t extBQ[2],strBQ[2];

                size_t extBM[2],strBM[2];

                DataBlock<T> BQ = Q.matrix_subspan(0, z, n, cz,extBQ,strBQ);
                DataBlock<T> BM = M.matrix_subspan(0, c, n,mc,extBM,strBM);

                size_t Cextt[2]= {cz,mc};
                size_t Cstrt[2]= {mc,1};

                DataBlock<T>  C(tempC,cz*mc,true,2,Cextt,Cstrt,false,false,false);


                size_t extBQT[2],strBQT[2];
                DataBlock<T> BQT=BQ.matrix_transpose(extBQT,strBQT);


                if(policy.should_use_gpu(BQT,BM,C,Math_Functions_Policy::default_cubic_treshold,1))
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(BQT,BM,C,policy.devicenum,true);
                else
                    In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(BQT,BM,C);

                size_t sexttt[2]= {n,mc};
                size_t sstrtt[2]= {mc,1};

                DataBlock<T>  S(tempS,n*mc,true,2,sexttt,sstrtt,false,false,false);


                switch (policy.algorithm_version)
                {
                case Math_MPI_Decomposition_Policy::Naive:
                    if(policy.should_use_gpu(BQ,C,S,Math_Functions_Policy::default_cubic_treshold,1))
                        GPU_Math_Functions<T>::matrix_multiply_dot_g(BQ,C,S,policy.devicenum,true);
                    else
                        In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(BQ,C,S);

                    break;
                case Math_MPI_Decomposition_Policy::Strassen:
                    strassen_multiply_h(BQ,C,S,false,false,policy);
                    break;
                case Math_MPI_Decomposition_Policy::WinogradVariant:
                    winograd_multiply_h(BQ,C,S,false,false,policy);
                }


                #pragma omp parallel for simd collapse(2)
                for (size_t i = 0; i < n; ++i)
                {
                    for (size_t j = c; j < n; ++j)
                    {
                        M(i, j) -= S(i, j-c);
                    }
                }
                z = c;
            }

            size_t vext[1],vstr[1];
            DataBlock<T> v = M.matrix_column(c,vext,vstr);

            for (size_t j = z; j < c; ++j)
            {
                size_t uext[1],ustr[1];
                DataBlock<T>  u = Q.matrix_column(j,uext,ustr);
                const T dot_pr =Math_Functions<T>::dot_product(u,v,&policy);

                #pragma omp parallel for simd
                for (size_t i = 0; i < n; ++i)
                {
                    v(i) -= dot_pr * u(i);
                }
            }

            // Normalize v
            const T norm = sqrt(Math_Functions<T>::dot_product(v,v,&policy));

            // Set column c of Q

            #pragma omp parallel for simd
            for (size_t i = 0; i < n; ++i)
            {
                Q(i,c) = v(i)/norm;
            }
        }


        // Compute R = Q^T * A
        size_t extQT[2],strQT[2];

        DataBlock<T> QT=Q.matrix_transpose(extQT,strQT);

        switch (policy.algorithm_version)
        {
        case Math_MPI_Decomposition_Policy::Naive:
            In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(QT,A,R);
            break;
        case Math_MPI_Decomposition_Policy::Strassen:
            strassen_multiply_h(QT,A,R,false,false,policy);
            break;
        case Math_MPI_Decomposition_Policy::WinogradVariant:
            winograd_multiply_h(QT,A,R,false,false,policy);
        }

        DataBlock_Host_Memory_Functions<T>::free_data_ptr(tempC,mm,policy.memmapped_files);
        DataBlock_Host_Memory_Functions<T>::free_data_ptr(tempS,nm,policy.memmapped_files);
        DataBlock_Host_Memory_Functions<T>::free_copy(M,policy.memmapped_files);

    }
}






template <typename T>
void Math_Functions_MPI<T>::MPI_recursive_multiplication_helper(const Math_MPI_RecursiveMultiplication_Policy *pol)
{
    const Math_MPI_RecursiveMultiplication_Policy policy = (pol != nullptr) ? *pol : get_default_policy();

    MPI_Status status;
    int message;
    for(;;)
    {
        MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, 0, policy.comm, &status);



        bool strassen=false;
        switch (message)
        {
        case Math_MPI_RecursiveMultiplication_Policy::Strassen:
            strassen=true;
        case Math_MPI_RecursiveMultiplication_Policy::WinogradVariant:
        {
            size_t problemsize;
            MPI_Recv(&problemsize, 1, mpi_get_type<size_t>(), MPI_ANY_SOURCE, 1, policy.comm, &status);
            DataBlock<T> A,B;
            bool ongpu=policy.should_use_gpu(problemsize,Math_Functions_Policy::default_cubic_treshold,false,7);
            bool separate_device_memory=false;
            if(ongpu)
            {
#if !defined(Unified_Shared_Memory)
                separate_device_memory=true;
#endif
            }

            A=DataBlock_MPI_Functions<T>::MPI_Recv_alloc_DataBlock(policy.memmapped_files,separate_device_memory,policy.devicenum,status.MPI_SOURCE, 2, policy.comm);
            B=DataBlock_MPI_Functions<T>::MPI_Recv_alloc_DataBlock(policy.memmapped_files,separate_device_memory,policy.devicenum,status.MPI_SOURCE, 3, policy.comm);


            bool crowm=true;
            size_t rowsC=A.dpextents[0],
                   colsC=B.dpextents[1];

            size_t extC[2];
            size_t strC[2];

            extC[0]=(crowm==true)?rowsC:colsC;
            extC[1]=(crowm==true)?colsC:rowsC;

            strC[0]=(crowm==true)? colsC:1;
            strC[1]=(crowm==true)?1: rowsC;

            T* C_data;
            size_t length=rowsC*colsC;
            if(separate_device_memory)
            {
                C_data=DataBlock_GPU_Memory_Functions<T>::alloc_data_device_ptr(length,policy.memmapped_files,policy.devicenum);
            }
            else
            {
                C_data=DataBlock_Host_Memory_Functions<T>::alloc_data_ptr(length,policy.memmapped_files);
            }

            DataBlock<T> C(C_data,length,crowm,2,extC,strC,false,false,separate_device_memory);


            if(policy.size_to_stop_recursion>=problemsize)
            {
                if(ongpu)
                {
                    GPU_Math_Functions<T>::matrix_multiply_dot_g(A, B, C,policy.devicenum,false);
                }
                else
                    In_Kernel_Mathfunctions<T>::matrix_multiply_dot_w(A, B, C);
            }
            else
            {
                if(strassen)
                    strassen_multiply_h(A,B,C,ongpu,separate_device_memory,policy);
                else
                    winograd_multiply_h(A,B,C,ongpu,separate_device_memory,policy);
            }

            DataBlock_MPI_Functions<T>::MPI_Send_DataBlock_pdata(C,status.MPI_SOURCE,4,policy.comm);

            DataBlock_MPI_Functions<T>::MPI_Free_DataBlock(A,policy.memmapped_files);
            DataBlock_MPI_Functions<T>::MPI_Free_DataBlock(B,policy.memmapped_files);
            if(separate_device_memory)
            {
                DataBlock_GPU_Memory_Functions<T>::free_data_device_ptr(C.dpdata,C.dpdatalength,policy.memmapped_files,policy.devicenum);
            }
            else
            {
                DataBlock_Host_Memory_Functions<T>::free_data_ptr(C.dpdata,C.dpdatalength,policy.memmapped_files);
            }


            break;
        }

        case Math_MPI_RecursiveMultiplication_Policy::End_Listener:
            goto endloop;
        }
    }

endloop:
    return;

}
template <typename T>
void Math_Functions_MPI<T>::MPI_recursion_helper_end(MPI_Comm pcomm)
{
    int commsize=0;
    MPI_Comm_size(pcomm, &commsize);
    int message=Math_MPI_RecursiveMultiplication_Policy::End_Listener;
    for (int i=0; i<commsize; i++)
    {
        MPI_Send(&message,1,MPI_INT,i,0,pcomm);
    }
}


#endif

